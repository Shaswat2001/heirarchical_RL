import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.spaces import Box
import sai_mujoco
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from agents.hiro.hiro import HIROAgent, HIRO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from networks.torch.mlp import DeterministicActor, Critic
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from trainers.hiro_trainer import HiroTrainer

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
env = gym.make("FrankaIkGolfCourseEnv-v0")
env = wrap_env(env)

goal_space = Box(low = -0.1, high= 0.1, shape=env.observation_space.shape, dtype= np.float32)
goal_observed_space = Box(
    low=np.concatenate([env.observation_space.low, goal_space.low]),
    high=np.concatenate([env.observation_space.high, goal_space.high]),
    dtype=np.float32
)

device = "cuda" if torch.cpu.is_available() else "cpu"

# instantiate a memory as experience replay
high_level_memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)
high_level_secondary_memory = RandomMemory(memory_size=3, num_envs=env.num_envs, device=device, replacement=False)
low_level_memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)

# instantiate the agent's models (function approximators).
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
high_level_models = {}
high_level_models["policy"] = DeterministicActor(env.observation_space, env.observation_space, device)
high_level_models["target_policy"] = DeterministicActor(env.observation_space, env.observation_space, device)
high_level_models["critic"] = Critic(env.observation_space, env.observation_space, device)
high_level_models["target_critic"] = Critic(env.observation_space, env.observation_space, device)

# initialize models' parameters (weights and biases)
for model in high_level_models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


low_level_models = {}
low_level_models["policy"] = DeterministicActor(goal_observed_space, env.action_space, device)
low_level_models["target_policy"] = DeterministicActor(goal_observed_space, env.action_space, device)
low_level_models["critic"] = Critic(goal_observed_space, env.action_space, device)
low_level_models["target_critic"] = Critic(goal_observed_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in low_level_models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = HIRO_DEFAULT_CONFIG.copy()

cfg["high_policy_sample_step"] = 3

agent = HIROAgent(high_models=high_level_models,
            low_models= low_level_models,
            high_memory= [high_level_memory, high_level_secondary_memory],
            low_memory= low_level_memory,
            cfg=cfg,
            observation_space=env.observation_space,
            goal_observed_space= goal_observed_space,
            action_space=env.action_space,
            goal_action_space= goal_space,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1500000, "headless": True}
trainer = HiroTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.single_agent_train()
