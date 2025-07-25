
import flax.linen
import flax.linen
import jax
import flax
import optax
import numpy as np
import jax.numpy as jnp

import copy
from utils.networks import GCTanhGaussianActor, GCValue, GCLaplaceActor
from typing import Any
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

RIS_CONFIG_DICT = {
    "agent_name": 'ris',  # Agent name.
    "lr": 1e-4,  # Learning rate.
    "batch_size": 1024,  # Batch size.
    "actor_hidden_dims": (256, 256),  # Actor network hidden dimensions.
    "value_hidden_dims": (256, 256),  # Value network hidden dimensions.
    "beta": 0.5, # Temperature in AWR.
    "layer_norm": False,  # Whether to use layer normalization.
    "tau": 0.005,
    "expectile_tau": 0.7,  # IQL expectile.
    "subgoal_step": 40, # step k to get subgoal
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "lambda": 0.1,
    "alpha": 0.1,
    "epsilon": 1e-16,
    "const_std": False,  # Whether to use constant standard deviation for the actor.
    "discrete": False,  # Whether the action space is discrete.
    # Dataset hyperparameters.
    "dataset_class": 'HGCDataset',  # Dataset class name.
    "value_p_curgoal": 0.2,  # Unused (defined for compatibility with GCDataset).
    "value_p_trajgoal": 0.5,  # Unused (defined for compatibility with GCDataset).
    "value_p_randomgoal": 0.3,  # Unused (defined for compatibility with GCDataset).
    "value_geom_sample": True,  # Unused (defined for compatibility with GCDataset).
    "actor_p_curgoal": 0.0,  # Probability of using the current state as the actor goal.
    "actor_p_trajgoal": 1.0,  # Probability of using a future state in the same trajectory as the actor goal.
    "actor_p_randomgoal": 0.0,  # Probability of using a random state as the actor goal.
    "actor_geom_sample": False,  # Whether to use geometric sampling for future actor goals.
    "gc_negative": True,  # Unused (defined for compatibility with GCDataset).
}

class RISAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def sample(self, distribution, rng):

        x_t = distribution.sample(seed = rng)
        action = jnp.tanh(x_t)
        log_prob = distribution.log_prob(x_t)
        log_prob -= jnp.log((1 - jnp.power(action, 2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdims=True)
        mean = jnp.tanh(distribution.mean())
        return action, log_prob, mean

    def critic_loss(self, batch, grad_params, rng = None):

        q = self.network.select("critic")(batch["observations"], batch["value_goals"], batch["actions"], params = grad_params).squeeze(-1)
        distribution = self.network.select("low_actor")(batch["next_observations"], batch["value_goals"])
        next_action, _, _ = self.sample(distribution, rng)
        target = self.network.select("target_critic")(batch["next_observations"], batch["value_goals"], next_action)
        target_q = batch["rewards"] + self.config["discount"]*batch["masks"]*target.squeeze(-1)
        # jax.debug.print("Q: {}", q.squeeze(-1))
        # jax.debug.print("Target Q: {}", target_q)
        critic_loss = ((target_q - q)**2).mean()
        jax.debug.print("Critic Loss: {}", critic_loss)
        critic_info = {
            "critic_loss" :  critic_loss,
            "q_mean" : q.mean(),
            "q_min" : q.min(),
            "q_max" : q.max()
        }

        return critic_loss, critic_info
        
    def high_actor_loss(self, batch, grad_params, rng=None):

        def value(state, goal):
            distribution = self.network.select("low_actor")(state, goal)
            _, _, action = self.sample(distribution, rng)
            V = self.network.select("critic")(state, goal, action)
            V = jnp.abs(jnp.clip(V, min = -100.0, max = 0.0))
            return V
        
        dist = self.network.select("high_actor")(batch["observations"], batch['high_actor_goals'], params=grad_params)

        # Compute target value
        new_subgoal = dist.loc
        policy_v_1 = value(batch["observations"], new_subgoal)
        policy_v_2 = value(new_subgoal, batch['high_actor_goals'])
        policy_v = jnp.maximum(policy_v_1, policy_v_2)

        # Compute subgoal distance loss
        v_1 = value(batch["observations"], batch['high_actor_targets'])
        v_2 = value(batch['high_actor_targets'], batch['high_actor_goals'])
        v = jnp.maximum(v_1, v_2)
        adv = - (v - policy_v)

        weight = flax.linen.softmax(adv/self.config["lambda"], axis=0).squeeze(-1)
        log_prob = dist.log_prob(batch['high_actor_targets']).sum(-1)
        actor_loss = - (log_prob * weight).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['high_actor_targets']) ** 2),
            'std': jnp.mean(dist.scale),
        }

        return actor_loss, actor_info
    
    def low_actor_loss(self, batch, grad_params, rng=None):

        dist = self.network.select('low_actor')(batch['observations'], batch['low_level_actor_goals'], params = grad_params)
        actions = dist.sample(seed=rng)
        
        actions_tanh = jnp.tanh(actions)

        # Q-value estimate
        q_val = self.network.select("critic")(
            batch["observations"], batch['low_level_actor_goals'], actions_tanh, params=grad_params
        )

        def sample_subgoals(state, goal, rng, num_subgoals=4):
            """Sample subgoals from high-level actor π^H(s_g | s, g)"""
            repeat_state = jnp.repeat(state[:, None, :], num_subgoals, axis=1)
            repeat_goal = jnp.repeat(goal[:, None, :], num_subgoals, axis=1)
            flat_state = repeat_state.reshape(-1, state.shape[-1])
            flat_goal = repeat_goal.reshape(-1, goal.shape[-1])
            dist = self.network.select("high_actor")(flat_state, flat_goal)
            subgoals = dist.sample(seed = rng)
            return subgoals.reshape(state.shape[0], num_subgoals, -1)
    
        subgoals = sample_subgoals(batch["observations"], batch["low_level_actor_goals"], rng)

        # Construct prior policy π(a|s, s_g) for each subgoal
        batch_size, num_subgoals = subgoals.shape[:2]
        expanded_obs = jnp.repeat(batch["observations"][:, None, :], num_subgoals, axis=1)
        flat_obs = expanded_obs.reshape(batch_size * num_subgoals, -1)
        flat_subgoals = subgoals.reshape(batch_size * num_subgoals, -1)
        prior_dists = self.network.select("target_low_actor")(flat_obs, flat_subgoals)
        expanded_actions = jnp.repeat(actions[:, None, :], num_subgoals, axis=1)
        flat_actions = expanded_actions.reshape(-1, expanded_actions.shape[-1])

        log_probs = prior_dists.log_prob(flat_actions)
        log_probs = log_probs.reshape(batch_size, num_subgoals, -1).sum(1)
        jax.debug.print("Actor loss: {}", log_probs.shape)
        # prior_log_prob_mean = jax.scipy.special.logsumexp(log_probs, axis=1) - jnp.log(num_subgoals)
        prior_log_prob_mean = jax.lax.stop_gradient(log_probs)
        # jax.debug.print("Prior value: {}", prior_log_probs)
        # KL divergence: log π(a|s,g) - log π_prior(a|s,g)
        log_prob = dist.log_prob(actions)
        # log_det_jacobian = 2 * (jnp.log(2) - actions - flax.linen.softplus(-2 * actions)).sum(-1)
        # log_prob_corrected = log_prob - log_det_jacobian
        kl_div = (log_prob - prior_log_prob_mean).sum(-1)

        jax.debug.print("KL divergence: {}", kl_div.shape)
        jax.debug.print("Q val: {}", kl_div)
        jax.debug.print("Q val: {}", q_val)
        # Final actor loss
        actor_loss = -jnp.mean(q_val.squeeze(-1) - self.config["alpha"] * kl_div)
        # jax.debug.print("Actor loss: {}", actor_loss)
        actor_info = {
            'actor_loss': actor_loss,
            'std': jnp.mean(dist.scale),
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):

        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'value/{k}'] = v

        rng, high_actor_rng = jax.random.split(rng)
        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        rng, low_actor_rng = jax.random.split(rng)
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = low_actor_loss + high_actor_loss + critic_loss
        return loss, info
    
    def update_target_network(self, network, function_name):
        new_target_params = jax.tree.map(
            lambda p, tp : p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{function_name}'],
            self.network.params[f'modules_target_{function_name}']
        )

        network.params[f'modules_target_{function_name}'] = new_target_params
    
    @jax.jit
    def update(self, batch):

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.update_target_network(new_network, "critic")
        self.update_target_network(new_network, "low_actor")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def get_actions(self, observation, goal= None, seed= None, temperature= 1.0):
        
        high_seed, low_seed = jax.random.split(seed)
        subgoal_dist = self.network.select('high_actor')(observation, goal, temperature=temperature)
        sub_goal = subgoal_dist.sample(seed= high_seed)

        dist = self.network.select('low_actor')(observation, sub_goal, temperature=temperature)
        actions = dist.sample(seed= low_seed)
        actions = jnp.tanh(actions)

        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)


        _cfg = copy.deepcopy(RIS_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]
        state_dim = ex_observations.shape[-1]

        high_actor_def = GCLaplaceActor(
            hidden_layers=_cfg['actor_hidden_dims'],
            action_dim=state_dim,
            const_std = _cfg["const_std"]
        )

        low_actor_def = GCTanhGaussianActor(
            hidden_layers=_cfg['actor_hidden_dims'],
            action_dim=action_dim,
            const_std = _cfg["const_std"]
        )

        target_low_actor_def = copy.deepcopy(low_actor_def)

        critic_def = GCValue(
            hidden_layers=_cfg["value_hidden_dims"],
            layer_norm=_cfg["layer_norm"]
        )

        target_critic_def = copy.deepcopy(critic_def)

        network_info = dict(
            high_actor=(high_actor_def, (ex_observations, ex_observations)),
            low_actor=(low_actor_def, (ex_observations, ex_observations)),
            target_low_actor=(target_low_actor_def, (ex_observations, ex_observations)),
            critic=(critic_def, (ex_observations, ex_observations, ex_actions)),
            target_critic=(target_critic_def, (ex_observations, ex_observations, ex_actions))
        )

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)
        network_tx = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=_cfg['lr'])
            )
        # network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_low_actor'] = params['modules_low_actor']

        return cls(rng, network=network, config=flax.core.FrozenDict(**_cfg))
    