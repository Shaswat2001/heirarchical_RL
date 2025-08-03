import jax
import flax
import optax
import copy
from typing import Any
import jax.numpy as jnp
from functools import partial
from utils.nf.networks import RealNVP, RealNVPEncoder, create_prior
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

NFGCBC_CONFIG_DICT = {
    "agent_name": 'nfgcbc',  # Agent name.
    "lr": 3e-4,  # Learning rate.
    "batch_size": 1024,  # Batch size.
    "actor_hidden_dims": (512, 512, 512),  # Actor network hidden dimensions.
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "clip_threshold": 100.0,
    "const_std": False,  # Whether to use constant standard deviation for the actor.
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
    "num_blocks": 12,
    "encode_dim": 100
}

class NFGCBCAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    prior: Any
    config: Any = nonpytree_field()

    @jax.jit
    def actor_loss(self, batch, grad_params, rng):

        obs_goal = jnp.concatenate([batch["observations"], batch["actor_goals"]], axis=-1).astype(jnp.float32)
        encod = self.network.apply("encoder")(obs_goal)
        z, logdets = self.network.apply("actor")(batch["actions"], encod, params=grad_params)
        loss = - (self.prior.log_prob(z) + logdets).mean()

        info = {
                    'actor/loss' : loss.item(),
                    'actor/logdets' : logdets.mean().item(),
                    'actor/norms/layer__0' : jnp.square(z).mean().item()
                }
        
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):

        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=['num_eval_episodes'])
    def get_actions(self, observation, goal, num_eval_episodes, seed= None):

        prior_sample = self.prior.sample(sample_shape=(num_eval_episodes,), seed=seed)
        obs_goal = jnp.concatenate([observation, goal], axis=-1).astype(jnp.float32)
        encode_obs_goal = self.network.select("encoder")(obs_goal)
        action, _ = self.network.apply("actor")(prior_sample, encode_obs_goal, reverse=True)

        return action

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        _cfg = copy.deepcopy(NFGCBC_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]
        observation_dim = ex_observations.shape[-1]
        encode_dim = _cfg["encode_dim"]

        nvp_def = RealNVP(
            _cfg["num_blocks"],
            _cfg["actor_hidden_dims"]
        )

        encode_def = RealNVPEncoder(
            input_size=observation_dim*2,
            out_size=encode_dim
        )

        network_info = dict(
            actor=(nvp_def, (action_dim, encode_dim)),
            encoder=(encode_def,(ex_observations*2))
        )

        prior = create_prior(action_dim)

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)

        network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, prior=prior, config=flax.core.FrozenDict(**_cfg))