import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

class GaussianActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_layers=[400, 300],
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        super().__init__(observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # Create MLP dynamically
        layers = []
        input_dim = self.num_observations
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden_net = nn.Sequential(*layers)
        self.action_layer = nn.Linear(input_dim, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.hidden_net(inputs["states"])
        return 2 * torch.tanh(self.action_layer(x)), self.log_std_parameter, {}
    
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_layers=[400, 300], clip_actions=False):
        super().__init__(observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        layers = []
        input_dim = self.num_observations
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden_net = nn.Sequential(*layers)
        self.action_layer = nn.Linear(input_dim, self.num_actions)

    def compute(self, inputs, role):
        x = self.hidden_net(inputs["states"])
        return 2 * torch.tanh(self.action_layer(x)), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_layers=[400, 300], clip_actions=False):
        super().__init__(observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Create MLP dynamically
        layers = []
        input_dim = self.num_observations + self.num_actions
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        x = self.hidden_net(x)
        return self.output_layer(x), {}
