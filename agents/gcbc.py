
import jax
import flax
import optax

import copy
import functools
from utils.networks import GCActor
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

GCBC_CONFIG_DICT = {
    "agent_name": 'gcbc',  # Agent name.
    "lr": 3e-4,  # Learning rate.
    "batch_size": 1024,  # Batch size.
    "actor_hidden_dims": (512, 512, 512),  # Actor network hidden dimensions.
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "const_std": True,  # Whether to use constant standard deviation for the actor.
    "discrete": False,  # Whether the action space is discrete.
    # Dataset hyperparameters.
    "dataset_class": 'GCDataset',  # Dataset class name.
    "value_p_curgoal": 0.0,  # Unused (defined for compatibility with GCDataset).
    "value_p_trajgoal": 1.0,  # Unused (defined for compatibility with GCDataset).
    "value_p_randomgoal": 0.0,  # Unused (defined for compatibility with GCDataset).
    "value_geom_sample": False,  # Unused (defined for compatibility with GCDataset).
    "actor_p_curgoal": 0.0,  # Probability of using the current state as the actor goal.
    "actor_p_trajgoal": 1.0,  # Probability of using a future state in the same trajectory as the actor goal.
    "actor_p_randomgoal": 0.0,  # Probability of using a random state as the actor goal.
    "actor_geom_sample": False,  # Whether to use geometric sampling for future actor goals.
    "gc_negative": True,  # Unused (defined for compatibility with GCDataset).
}

class GCBCAgent(flax.struct.PyTreeNode):

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)


        _cfg = copy.deepcopy(GCBC_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]

        actor_def = GCActor(
            hidden_dims=_cfg['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=_cfg['const_std'],
        )

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_observations))
        )

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)
        network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**_cfg))