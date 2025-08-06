import jax
import flax
import optax
import copy
from typing import Any
import jax.numpy as jnp
from functools import partial
from utils.nf.networks import RealNVP, RealNVPEncoder, create_prior
from utils.networks import GCValue
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

NFHIQL_CONFIG_DICT = {
    "agent_name": 'nfgciql',  # Agent name.
    "lr": 3e-4,  # Learning rate.
    "batch_size": 256,  # Batch size.
    "actor_hidden_dims": (512, 512, 512),  # Actor network hidden dimensions.
    "value_hidden_dims": (256, 256),  # Value network hidden dimensions.
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "tau": 0.005,
    "clip_threshold": 100,
    "beta": 0.3, # Temperature in AWR.
    "expectile_tau": 0.9,  # IQL expectile.
    "subgoal_step": 40, # step k to get subgoal
    "layer_norm": True,  # Whether to use layer normalization.
    "discount": 0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
    "const_std": True,  # Whether to use constant standard deviation for the actor.
    "discrete": False,  # Whether the action space is discrete.
    # Dataset hyperparameters.
    "dataset_class": 'HGCDataset',  # Dataset class name.
    "value_p_curgoal": 0.2,  # Unused (defined for compatibility with GCDataset).
    "value_p_trajgoal": 0.5,  # Unused (defined for compatibility with GCDataset).
    "value_p_randomgoal": 0.3,  # Unused (defined for compatibility with GCDataset).
    "value_geom_sample": True,  # Unused (defined for compatibility with GCDataset).
    "actor_p_curgoal": 1.0,  # Probability of using the current state as the actor goal.
    "actor_p_trajgoal": 0.0,  # Probability of using a future state in the same trajectory as the actor goal.
    "actor_p_randomgoal": 0.0,  # Probability of using a random state as the actor goal.
    "actor_geom_sample": True,  # Whether to use geometric sampling for future actor goals.
    "gc_negative": True,  # Unused (defined for compatibility with GCDataset).
    "num_blocks": 6,
    "encode_dim": 64,
    "channels": 256, 
    "noise_std": 0.1,
    "weight_decay": 1e-6

}

class NFHIQLAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    high_prior: Any
    low_prior: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):

        val = jnp.where(adv >= 0, expectile, (1 - expectile))
        return val * (diff**2)

    @jax.jit
    def value_loss(self, batch, grad_params):
        
        next_v = self.network.select("target_value")(batch["next_observations"], batch["value_goals"])
        
        target_v = batch["rewards"] + self.config["discount"]*batch["masks"]*next_v

        v = self.network.select("target_value")(batch["observations"], batch["value_goals"])

        adv = target_v - v

        v_c = self.network.select("value")(batch["observations"], batch["value_goals"], params=grad_params)

        value_loss = self.expectile_loss(adv, target_v - v_c, self.config["expectile_tau"]).mean()

        value_info = {
            "value_loss" :  value_loss,
            "v_mean" : v_c.mean(),
            "v_max" :  v_c.max(),
            "v_min" :  v_c.min(),
        }

        return value_loss, value_info
    
    @jax.jit
    def high_actor_loss(self, batch, grad_params, rng=None):

        v = self.network.select("value")(batch["observations"], batch["high_actor_goals"])
        next_v = self.network.select("value")(batch["high_actor_targets"], batch["high_actor_goals"])

        adv = next_v - v
        exp_adv = jnp.minimum(jnp.exp(self.config["beta"]*adv),100)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['high_actor_targets'])

        actor_loss = -(exp_adv*log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['high_actor_targets']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info
    
    @jax.jit
    def high_actor_loss(self, batch, grad_params, rng):

        v = self.network.select("value")(batch["observations"], batch["high_actor_goals"])
        next_v = self.network.select("value")(batch["high_actor_targets"], batch["high_actor_goals"])

        adv = next_v - v
        exp_adv = jnp.minimum(jnp.exp(self.config["beta"]*adv),100).squeeze(1)

        obs_goal = jnp.concatenate([batch["observations"], batch["high_actor_goals"]], axis=-1).astype(jnp.float32)
        encod = self.network.select("high_encoder")(obs_goal, params=grad_params)
        z, logdets = self.network.select("high_actor")(batch['high_actor_targets'], encod, params=grad_params)
        loss = -(exp_adv*(self.high_prior.log_prob(z) + logdets)).mean()
        entropy, mse = self.get_high_entropy(batch, rng)

        info = {
            'actor_loss': loss,
            'adv': adv.mean(),
            'actor_logdets' : logdets.mean(),
            'actor_norms_layer__0' : jnp.square(z).mean(),
            'entropy': entropy,
            'entropy': entropy,
            'mse': mse
        }
        
        return loss, info
    
    @jax.jit
    def low_actor_loss(self, batch, grad_params, rng=None):

        v = self.network.select("value")(batch["observations"], batch["low_level_actor_goals"])
        next_v = self.network.select("value")(batch["next_observations"], batch["low_level_actor_goals"])

        adv = next_v - v
        exp_adv = jnp.minimum(jnp.exp(self.config["beta"]*adv),100).squeeze(1)

        obs_goal = jnp.concatenate([batch["observations"], batch["low_level_actor_goals"]], axis=-1).astype(jnp.float32)
        encod = self.network.select("low_encoder")(obs_goal, params=grad_params)
        z, logdets = self.network.select("low_actor")(batch['actions'], encod, params=grad_params)
        loss = -(exp_adv*(self.low_prior.log_prob(z) + logdets)).mean()
        entropy, mse = self.get_low_entropy(batch, rng)

        info = {
            'actor_loss': loss,
            'adv': adv.mean(),
            'actor_logdets' : logdets.mean(),
            'actor_norms_layer__0' : jnp.square(z).mean(),
            'entropy': entropy,
            'entropy': entropy,
            'mse': mse
        }

        return loss, info
    
    @jax.jit
    def get_low_entropy(self, batch, seed=None):

        obs_goal = jnp.concatenate([batch["observations"], batch["low_level_actor_goals"]], axis=-1).astype(jnp.float32)
        prior_sample = self.low_prior.sample(sample_shape=(obs_goal.shape[0],), seed=seed)
        encode_obs_goal = self.network.select("low_encoder")(obs_goal)
        p_actions, p_logdets = self.network.select("low_actor")(prior_sample, encode_obs_goal, reverse=True)
        entropy = (self.low_prior.log_prob(prior_sample) - p_logdets).mean() 
        mse = ((p_actions - batch["actions"] ) **2).mean()
        return entropy, mse
    
    @jax.jit
    def get_high_entropy(self, batch, seed=None):

        obs_goal = jnp.concatenate([batch["observations"], batch["high_actor_goals"]], axis=-1).astype(jnp.float32)
        prior_sample = self.high_prior.sample(sample_shape=(obs_goal.shape[0],), seed=seed)
        encode_obs_goal = self.network.select("high_encoder")(obs_goal)
        p_actions, p_logdets = self.network.select("high_actor")(prior_sample, encode_obs_goal, reverse=True)
        entropy = (self.high_prior.log_prob(prior_sample) - p_logdets).mean() 
        mse = ((p_actions - batch["high_actor_targets"] ) **2).mean()
        return entropy, mse

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):

        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, high_actor_rng = jax.random.split(rng)
        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        rng, low_actor_rng = jax.random.split(rng)
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = low_actor_loss + value_loss + high_actor_loss
        info["total_loss"] = loss
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
        self.update_target_network(new_network, "value")

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=['num_eval_episodes'])
    def get_actions(self, observation, goal, num_eval_episodes=None, seed=None):

        prior_sample = self.high_prior.sample(sample_shape=(num_eval_episodes,), seed=seed)
        obs_goal = jnp.concatenate([observation, goal], axis=-1).astype(jnp.float32)
        encode_obs_goal = self.network.select("high_encoder")(obs_goal)
        sub_goal, _ = self.network.select("high_actor")(prior_sample, encode_obs_goal, reverse=True)

        prior_sample = self.low_prior.sample(sample_shape=(num_eval_episodes,), seed=seed)
        obs_goal = jnp.concatenate([observation, sub_goal], axis=-1).astype(jnp.float32)
        encode_obs_goal = self.network.select("low_encoder")(obs_goal)
        action, _ = self.network.select("low_actor")(prior_sample, encode_obs_goal, reverse=True)
        action = jnp.clip(action, -1.0, 1.0)

        return action

    # @partial(jax.jit, static_argnames=['num_eval_episodes'])
    # def get_denoised_action(self, observation, goal, num_eval_episodes=None, seed= None):
         
    #     def log_prob_fn(x, y):
    #         z, logdets = self.network.select("actor")(x, y)
    #         logprob = self.prior.log_prob(z) + logdets
    #         return logprob.sum()
        
    #     prior_sample = self.prior.sample(sample_shape=(num_eval_episodes,), seed= seed)
    #     observation_goal = jnp.concatenate([observation, goal], axis=-1).astype(jnp.float32)
    #     observation_goal_z = self.network.select("encoder")(observation_goal)
    #     action, _ = self.network.select("actor")(
    #         prior_sample, 
    #         observation_goal_z, 
    #         reverse=True, 
    #         )
        
    #     action = jax.lax.stop_gradient(action)
    #     action_score = jax.grad(log_prob_fn)(action, observation_goal_z)
    #     action = action + self.config["noise_std"]**2 * action_score
    #     return action


    @classmethod
    def create(cls, seed, ex_observations, ex_actions, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        _cfg = copy.deepcopy(NFHIQL_CONFIG_DICT)
        _cfg.update(cfg if cfg is not None else {})

        action_dim = ex_actions.shape[-1]
        observation_dim = ex_observations.shape[-1]
        encode_dim = _cfg["encode_dim"]


        low_network_dims = (action_dim+encode_dim, _cfg["channels"]+encode_dim, _cfg["channels"], action_dim//2)
        high_network_dims = (observation_dim+encode_dim, _cfg["channels"]+encode_dim, _cfg["channels"], observation_dim//2)

        low_nvp_def = RealNVP(
            _cfg["num_blocks"],
            action_dim,
            low_network_dims
        )

        low_encode_def = RealNVPEncoder(
            input_size=observation_dim*2,
            out_size=encode_dim
        )

        high_nvp_def = RealNVP(
            _cfg["num_blocks"],
            observation_dim,
            high_network_dims
        )

        high_encode_def = RealNVPEncoder(
            input_size=observation_dim*2,
            out_size=encode_dim
        )

        value_def = GCValue(
            hidden_layers=_cfg["value_hidden_dims"],
            layer_norm=_cfg["layer_norm"]
        )

        target_value_def = copy.deepcopy(value_def)

        network_info = dict(
            low_actor=(low_nvp_def, (ex_actions, jnp.zeros((1, encode_dim)))),
            low_encoder=(low_encode_def,(jnp.concatenate([ex_observations, ex_observations], axis=-1))),
            high_actor=(high_nvp_def, (ex_observations, jnp.zeros((1, encode_dim)))),
            high_encoder=(high_encode_def,(jnp.concatenate([ex_observations, ex_observations], axis=-1))),
            value=(value_def, (ex_observations, ex_observations)),
            target_value=(target_value_def, (ex_observations, ex_observations))
        )

        high_prior = create_prior(observation_dim)
        low_prior = create_prior(action_dim)

        network = {k: v[0] for k,v in network_info.items()}
        network_args = {k: v[1] for k,v in network_info.items()}

        network_def = ModuleDict(network)

        lr_scheduler = optax.warmup_cosine_decay_schedule(
                    init_value=0.0, peak_value=_cfg["lr"],
                    warmup_steps=500, decay_steps=1000000,
                    end_value=1e-6,
        )
        network_tx = optax.adamw(learning_rate=lr_scheduler, weight_decay=_cfg["weight_decay"])

        # network_tx = optax.adam(learning_rate=_cfg['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, high_prior=high_prior, low_prior=low_prior, config=flax.core.FrozenDict(**_cfg))