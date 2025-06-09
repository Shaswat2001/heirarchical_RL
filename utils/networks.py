import flax.linen as nn

from typing import Any, Sequence

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

class MLP:

    hidden_layers: Sequence[int]
    activation: nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        
        for i, size in enumerate(self.hidden_layers):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_layers) or self.activate_final:
                x = self.activation(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_layers) - 2:
                self.sow('intermediates', 'feature', x)
        return x