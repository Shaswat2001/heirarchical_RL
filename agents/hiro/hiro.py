from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch.ddpg import DDPG
from skrl.agents.torch.base import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model

HIRO_DEFAULT_CONFIG = {
    "high_policy_sample_step": 1,
}

class HighLevelDDPG(DDPG):

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        goal_action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        
        _cfg = {}
        _cfg.update(cfg if cfg is not None else {})

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        self.goal_action_space = goal_action_space
        self.scale = torch.Tensor(self.goal_action_space.high)

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        Agent.init(self, trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        assert len(self.secondary_memories) == 1

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

        self.secondary_memories[0].create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
        self.secondary_memories[0].create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
        self.secondary_memories[0].create_tensor(name="goals", size=self.observation_space, dtype=torch.float32)
        self.secondary_memories[0].create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.secondary_memories[0].create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.secondary_memories[0].create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.secondary_memories[0].create_tensor(name="truncated", size=1, dtype=torch.bool)

        self._secondary_tensors_names = ["states", "goals", "actions", "rewards", "next_states", "terminated", "truncated"]
        # clip noise bounds
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

    def record_transition(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        low_level_policy,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param goals: Actions taken by the high level agent
        :type goals: torch.Tensor
        :param actions: Actions taken by the low level agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        assert len(self.secondary_memories) == 1
        Agent.record_transition(
            self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        self.secondary_memories[0].add_samples(states=states,
                                               goals= goals,
                                               actions=actions,
                                               rewards=rewards,
                                               next_states=next_states,
                                               terminated=terminated,
                                               truncated=truncated
                                               )

        if self.secondary_memories[0].filled:
            (
                sampled_states,
                sampled_goals,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.secondary_memories[0].sample_all(names=self._secondary_tensors_names)[0]

            self.secondary_memories[0].reset()

            (
                states, 
                goals, 
                rewards, 
                next_states, 
                terminated, 
                truncated
            ) = self.off_policy_correction(states= sampled_states,
                                           goals= sampled_goals,
                                           actions= sampled_actions,
                                           rewards= sampled_rewards,
                                           next_states= sampled_next_states,
                                           terminated= sampled_terminated,
                                           truncated= sampled_truncated,
                                           low_level_policy=low_level_policy)
            
            self.memory.add_samples(states=states,
                                    actions=goals,
                                    rewards=rewards,
                                    next_states=next_states,
                                    terminated=terminated,
                                    truncated=truncated)

    def sample(self, 
               state: torch.Tensor, 
               next_state: torch.Tensor, 
               action: torch.Tensor, 
               states: torch.Tensor,
               actions: torch.Tensor, 
               low_level_policy,
               k: int= 8):
        
        diff_goal = (next_state - state).reshape(-1, state.shape[0])

        original_goal = action[0,:]
        original_goal = original_goal.reshape(-1, original_goal.shape[0])
        random_goals = torch.normal(mean=diff_goal.expand(k,-1), std=.5*self.scale[None, :])
        random_goals = random_goals.clip(-self.scale, self.scale)

        candidates = torch.concat([original_goal, diff_goal, random_goals], dim=0)

        # For ease
        seq_len = states.shape[0]
        action_dim = actions.shape[-1]
        obs_dim = state.shape[0]
        ncands = candidates.shape[0]

        true_actions = actions
        observations = states
        goal_shape = (seq_len, obs_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = torch.zeros((ncands,seq_len, action_dim))
        for c in range(ncands):
            subgoal = candidates[c]
            candidate = (subgoal + states[:, :obs_dim]) - states[:, :obs_dim]
            candidate = candidate.reshape(*goal_shape)
            combined_state = torch.concat([observations, candidate], dim= 1)
            policy_actions[c] = low_level_policy.policy.act({"states": low_level_policy._state_preprocessor(combined_state)}, role="policy")[0]

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, seq_len, action_dim)).transpose(0, 1, 2)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[max_indices,:]

    
    def off_policy_correction(self,
                              states: torch.Tensor,
                              goals: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              next_states: torch.Tensor,
                              terminated: torch.Tensor,
                              truncated: torch.Tensor,
                              low_level_policy):
        
        
        corrected_states = states[0,:]
        corrected_next_states = next_states[-1,:]
        currected_terminated = terminated[-1,:]
        currected_truncated = truncated[-1,:]
        corrected_rewards = torch.sum(rewards)

        corrected_goal = self.sample(corrected_states, corrected_next_states, goals, states, actions, low_level_policy)

        return corrected_states, corrected_goal, corrected_rewards, corrected_next_states, currected_terminated, currected_truncated 

class HIROAgent:

    def __init__(
        self,
        high_models: Mapping[str, Model],
        low_models: Mapping[str, Model],
        high_memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        low_memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        goal_observed_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        goal_action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        
        _cfg = copy.deepcopy(HIRO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        self.high_agent = HighLevelDDPG(models= high_models,
                          memory= high_memory,
                          observation_space= observation_space,
                          action_space= action_space,
                          goal_action_space= goal_action_space,
                          device= device,
                          cfg= _cfg)
        
        self.low_agent = DDPG(models= low_models,
                          memory= low_memory,
                          observation_space= goal_observed_space,
                          action_space= action_space,
                          device= device,
                          cfg= _cfg)
        
        self._high_policy_sample_step = cfg["high_policy_sample_step"]

    def act(self, states: torch.Tensor, goals: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using HIRO

        :param states: Environment's states
        :type states: torch.Tensor
        :param goals: Input to low level policy
        :type goals: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        
        
        if timestep % self._high_policy_sample_step == 0:
            goal, _, _ = self.high_agent.act(states = states,timestep= timestep, timesteps = timesteps)
        else:
            goal = goals
        

        low_level_states = torch.concat([states, goal], dim= -1)
        actions, _, outputs = self.low_agent.act(states = low_level_states,timestep= timestep, timesteps = timesteps)
        
        return goal, actions, None, outputs
    
    def record_high_transition(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        
        self.high_agent.record_transition(states, 
                                          goals,
                                          actions, 
                                          rewards, 
                                          next_states, 
                                          terminated, 
                                          truncated, 
                                          self.low_agent,
                                          infos, 
                                          timestep, 
                                          timesteps)
        
    def record_low_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        
        self.low_agent.record_transition(states, 
                                          actions, 
                                          rewards, 
                                          next_states, 
                                          terminated, 
                                          truncated, 
                                          infos, 
                                          timestep, 
                                          timesteps)

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:

        self.high_agent.init(trainer_cfg)
        self.low_agent.init(trainer_cfg)
    
    def set_running_mode(self, mode: str):

        self.high_agent.set_running_mode(mode)
        self.low_agent.set_running_mode(mode)

    def pre_interaction(self, timestep: int, timesteps: int):
        
        self.high_agent.pre_interaction(timestep= timestep, timesteps= timesteps)
        self.low_agent.pre_interaction(timestep= timestep, timesteps= timesteps)
    
    def post_interaction(self, timestep: int, timesteps: int):

        if timestep % self._high_policy_sample_step == 0 and self.high_agent.memory.memory_index > 0:
            self.high_agent.post_interaction(timestep= timestep, timesteps= timesteps)
        
        self.low_agent.post_interaction(timestep= timestep, timesteps= timesteps)

    def track_data(self, tag: str, value: float):

        self.high_agent.track_data(tag= tag, value= value)
        self.low_agent.track_data(tag= tag, value= value)