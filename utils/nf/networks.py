import distrax
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from flax import struct
from typing import Any, Sequence, Optional

def create_prior(input_dim):

    loc = jnp.zeros(input_dim, dtype=jnp.float32)
    cov = jnp.eye(input_dim, dtype=jnp.float32)
    return distrax.MultivariateNormalFullCovariance(loc=loc, covariance_matrix= cov)

def kernel_init(key, shape, dtype=jnp.float32):
    in_features = shape[0]
    k = jnp.sqrt( 1.0 / in_features )
    return jax.random.uniform(key, shape, dtype, minval=-k, maxval=k)

def bias_init(key, shape, dtype, in_features):
    k = jnp.sqrt( 1.0 / in_features )
    return jax.random.uniform(key, shape, dtype, minval=-k, maxval=k)

class MLP(nn.Module):

    hidden_layers: Sequence[int]
    activation: Any = nn.relu
    activate_final: bool = False
    kernel_init: Any = kernel_init
    bias_init: Any = bias_init
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        
        for i, size in enumerate(self.hidden_layers):
            if i==0:
                continue
            if i == len(self.hidden_layers)-1:
                x = nn.Dense(size, kernel_init=jax.nn.initializers.zeros)(x)
            else:
                x = nn.Dense(size, kernel_init=self.kernel_init, bias_init=partial(self.bias_init, in_features=self.hidden_layers[i-1]))(x)
            if i + 1 < len(self.hidden_layers) or self.activate_final:
                x = self.activation(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                
            if i == len(self.hidden_layers) - 2:
                self.sow('intermediates', 'feature', x)
        return x

class PLU(nn.Module):
    features: int
    key: jax.Array = struct.field(default_factory=lambda: jax.random.PRNGKey(0))

    def setup(self):
        d = self.features
        key = self.key
        
        w_shape = (d, d)
        w_init = nn.initializers.orthogonal()(key, w_shape)
        P, L, U = jax.scipy.linalg.lu(w_init)
        s = jnp.diag(U)
        U = U - jnp.diag(s)

        self.P = P
        self.P_inv = jax.scipy.linalg.inv(P)
        
        self.L_init = jnp.tril(L, k=-1)
        self.U_init = jnp.triu(U, k=1)
        self.s_init = s

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        d = self.features

        L_free = self.param("L", lambda rng: self.L_init)
        U_free = self.param("U", lambda rng: self.U_init)
        
        L = jnp.tril(L_free, k=-1) + jnp.eye(d)
        U = jnp.triu(U_free, k=1)
        s = self.param("s", lambda rng: self.s_init)
        
        W = self.P @ L @ (U + jnp.diag(s))

        if not reverse:
            z = jnp.dot(x, W)
            logdet = jnp.sum(jnp.log(jnp.abs(s)))
            return z , jnp.expand_dims( logdet, 0)
        
        else:
            
            U_inv = jax.scipy.linalg.solve_triangular(U + jnp.diag(s), jnp.eye(self.features), lower=False)
            L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(self.features), lower=True, unit_diagonal=True)
            
            W_inv = U_inv @ L_inv @ self.P_inv
            
            z = jnp.dot(x, W_inv)
            logdet = jnp.sum(jnp.log(jnp.abs(s)))
            return z, -jnp.expand_dims( logdet, 0)
    
class CouplingLayer(nn.Module):
    
    input_shape: int
    hidden_layers: Sequence[int]

    def setup(self):
        
        self.l = PLU(features=self.input_shape)
        self.s = MLP(self.hidden_layers, activation= nn.leaky_relu, activate_final= False, layer_norm= True)
        self.t = MLP(self.hidden_layers, activation= nn.leaky_relu, activate_final= False, layer_norm= True)

    def __call__(self, x, y, reverse= False):
        
        if reverse:
            return self.reverse(x, y)
        else:
            return self.forward(x, y)
    
    def forward(self, x, y):
        
        x, log_det = self.l(x)
        
        x1, x2 = jnp.array_split(x, 2, axis=1)
        s = self.s(jnp.concatenate([x1, y], axis=-1))
        t = self.t(jnp.concatenate([x1, y], axis=-1))
        
        x2 = (x2 - t) * jnp.exp(-s)
        x = jnp.concatenate([x1, x2], axis=1)
        log_det += -jnp.sum(s, axis=1)

        return x, log_det


    def reverse(self, z, y):
        
        z1, z2 = jnp.array_split(z, 2, axis=1)
        s = self.s(jnp.concatenate([z1, y], axis=-1))
        t = self.t(jnp.concatenate([z1, y], axis=-1))

        z2 = z2 * jnp.exp(s) + t
        z = jnp.concatenate([z1, z2], axis=1)

        z, log_det = self.l(z, reverse=True)

        log_det += jnp.sum(s, axis=1)

        return z, log_det

class RealNVP(nn.Module):

    num_blocks: int
    input_shape: int
    hidden_layers: Sequence[int]

    def setup(self):
        
        self.blocks = [
            CouplingLayer(self.input_shape, self.hidden_layers)
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