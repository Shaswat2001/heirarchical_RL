import distrax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any, Sequence, Optional
from 

def create_prior(input_dim):

    loc = jnp.zeros(input_dim, dtype=jnp.float32)
    cov = jnp.eye(input_dim, dtype=jnp.float32)
    return distrax.MultivariateNormalFullCovariance(loc=loc, covariance_matrix= cov)

class CouplingLayer(nn.Module):
     
    def setup(self):
        pass

class RealNVP(nn.Module):

    pass

class RealNVPEncoder(nn.Module):

    pass