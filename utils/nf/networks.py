import distrax
import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any, Sequence, Optional
from utils.networks import MLP

def create_prior(input_dim):

    loc = jnp.zeros(input_dim, dtype=jnp.float32)
    cov = jnp.eye(input_dim, dtype=jnp.float32)
    return distrax.MultivariateNormalFullCovariance(loc=loc, covariance_matrix= cov)

class CouplingLayer(nn.Module):

    hidden_layers: Sequence[int]

    def setup(self):
        
        self.s = MLP(self.hidden_layers, activation= nn.leaky_relu, activate_final= False, layer_norm= True)
        self.t = MLP(self.hidden_layers, activation= nn.leaky_relu, activate_final= False, layer_norm= True)

    def __call__(self, x, y, reverse= False):
        
        if reverse:
            return self.reverse(x, y)
        else:
            return self.forward(x, y)
    
    def forward(self, x, y):
        
        x1, x2 = jnp.hsplit(x, 2)
        s = self.s(jnp.concatenate([x1, y], axis=-1))
        t = self.t(jnp.concatenate([x1, y], axis=-1))

        x2 = (x2 - t) * jnp.exp(-s)
        x = jnp.concatenate([x1, x2], axis=1)
        log_det = -jnp.sum(s, axis=1)

        return x, log_det


    def reverse(self, z, y):
        
        z1, z2 = jnp.hsplit(z, 2)
        s = self.s(jnp.concatenate([z1, y], axis=-1))
        t = self.t(jnp.concatenate([z1, y], axis=-1))

        z2 = z2 * jnp.exp(s) + t
        z = jnp.concatenate([z1, z2], axis=1)
        log_det = jnp.sum(s, axis=1)

        return z, log_det

class RealNVP(nn.Module):

    num_blocks: int
    hidden_layers: Sequence[int]

    def setup(self):
        
        self.blocks = [
            CouplingLayer(self.hidden_layers)
            for _ in range(self.num_blocks)
        ]
    
    def __call__(self, x, y, reverse=False):
        if reverse:
            return self.reverse(x, y)
        else:
            return self.forward(x, y)

    def forward(self, x, y):
        log_dets = jnp.zeros(x.shape[0], dtype=x.dtype)
        for block in self.blocks:
            x, log_det = block(x, y)
            log_dets = log_dets + log_det
        return x, log_dets
        
    def reverse(self, x, y):
        log_dets = jnp.zeros(x.shape[0], dtype=x.dtype)
        for block in reversed(self.blocks):
            x, log_det = block(x, y, reverse=True)
            log_dets = log_dets + log_det
        return x, log_dets

class RealNVPEncoder(nn.Module):

    input_size: int
    out_size: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(512)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = jax.nn.silu(x)

        x = nn.Dense(512)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = jax.nn.silu(x)

        x = nn.Dense(512)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = jax.nn.silu(x)

        x = nn.Dense(512)(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = jax.nn.silu(x)

        # Final projection to rep_size
        x = nn.Dense(self.out_size)(x)
        return x