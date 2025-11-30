import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class



@register_expert_class("mlp.full", lambda batch_size, hid_dim: torch.empty((batch_size, 784)))
class MLP(nn.Module):
    """
    A simple feedforward neural network with two hidden layers.

    Args:
        num_classes (int): Number of output classes
    """

    def __init__(self, num_classes: int = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@register_expert_class("mlp.head", lambda batch_size, hid_dim: torch.empty((batch_size, 784)))
class MLPFront(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

@register_expert_class("mlp.tail", lambda batch_size, hid_dim: torch.empty((batch_size, 200)))
class MLPBack(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
