
import jax
import flax
import optax
import jax.numpy as jnp

import copy
import functools
from utils.networks import GCActor, GCDetActor
from typing import Any, Sequence
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

HBC_CONFIG_DICT = {
    "agent_name": 'hbc',  # Agent name.
    "lr": 1e-4,  # Learning rate.
    "batch_size": 1024,  # Batch size.
    "actor_hidden_dims": (512, 512, 512),  # Actor network hidden dimensions.
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "clip_threshold": 10.0,
    "subgoal_step": 10, # step k to get subgoal
    "const_std": True,  # Whether to use constant standard deviation for the actor.
    "discrete": False,  # Whether the action space is discrete.
    # Dataset hyperparameters.
    "dataset_class": 'HGCDataset',  # Dataset class name.
    "value_p_curgoal": 0.2,  # Unused (defined for compatibility with GCDataset).
    "value_p_trajgoal": 0.5,  # Unused (defined for compatibility with GCDataset).
    "value_p_randomgoal": 0.3,  # Unused (defined for compatibility with GCDataset).
    "value_geom_sample": True,  # Unused (defined for compatibility with GCDataset).
    "actor_p_curgoal": 0.2,  # Probability of using the current state as the actor goal.
    "actor_p_trajgoal": 0.5,  # Probability of using a future state in the same trajectory as the actor goal.
    "actor_p_randomgoal": 0.3,  # Probability of using a random state as the actor goal.
    "actor_geom_sample": True,  # Whether to use geometric sampling for future actor goals.
    "gc_negative": True,  # Unused (defined for compatibility with GCDataset).
    "bc_method": "mse"
}

class HBCAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    weights: Sequence[int] = (1000, 1000, 1000, 1000, 1000, 1000, 1)

    def high_actor_loss(self, batch, grad_params, rng=None):

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['high_actor_targets'])

        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }

        actor_info.update(
            {
                'mse': jnp.mean((dist.mode() - batch['high_actor_targets']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        )

        return actor_loss, actor_info
    
    def low_actor_loss(self, batch, grad_params, rng=None):

        dist = self.network.select('low_actor')(batch['observations'], batch['low_level_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }

        actor_info.update(
            {
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        )

        return actor_loss, actor_info
    
    def low_actor_mse_loss(self, batch, grad_params, rng=None):

        actions = self.network.select('low_actor')(batch['observations'], batch['low_level_actor_goals'], params=grad_params)

        actor_loss = optax.huber_loss(actions, batch['actions']) * self.weights
        actor_loss = actor_loss.mean() 
        # actor_loss = optax.huber_loss(actions, batch['actions']).mean()

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):

        info = {}
        rng = rng if rng is not None else self.rng

        rng, high_actor_rng = jax.random.split(rng)
        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)

        rng, low_actor_rng = jax.random.split(rng)
        if self.config["bc_method"] == "mse":
            low_actor_loss, low_actor_info = self.low_actor_mse_loss(batch, grad_params, low_actor_rng)
        else:
            low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = low_actor_loss + high_actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    def get_actions(self, observation, goal= None, seed= None, temperature= 1.0):

        high_seed, low_seed = jax.random.split(seed)
        subgoal_dist = self.network.select('high_actor')(observation, goal, temperature=temperature)
        sub_goal = subgoal_dist.sample(seed= high_seed)

        if self.config["bc_method"] == "mse":
            actions = self.network.select('low_actor')(observation, sub_goal)
        else:
            dist = self.network.select('low_actor')(observation, sub_goal, temperature=temperature)
            actions = dist.sample(seed= low_seed)
            actions = jnp.clip(actions, -1.0, 1.0)

        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        _cfg = copy.deepcopy(HBC_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]
        state_dim = ex_observations.shape[-1]

        high_actor_def = GCActor(
            hidden_layers=_cfg['actor_hidden_dims'],
            action_dim=state_dim,
        )
        
        if _cfg["bc_method"] == "mse":
            low_actor_def = GCDetActor(
                hidden_layers=_cfg['actor_hidden_dims'],
                action_dim=action_dim,
            )
        else:
            low_actor_def = GCActor(
                hidden_layers=_cfg['actor_hidden_dims'],
                action_dim=action_dim,
            )

        network_info = dict(
            high_actor=(high_actor_def, (ex_observations, ex_observations)),
            low_actor=(low_actor_def, (ex_observations, ex_observations))
        )

        weights = jnp.array((1000, 1000, 1000, 1000, 1000, 1000, 1)).reshape(1,-1)

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)
        network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**_cfg), weights = weights)