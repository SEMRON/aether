import torch

from hivemind.dht import DHT


class DecentralizedOptimizerBase(torch.optim.Optimizer):
    """A shared interface for all hivemind optimizers. Cooperates with DHT peers to train a shared model"""

    def __init__(self, optimizer: torch.optim.Optimizer, dht: DHT):
        self.optimizer, self.dht = optimizer, dht

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, param_group: dict) -> None:
        raise ValueError(
            f"{self.__class__.__name__} does not support calling add_param_group after creation."
            f"Please provide all parameter groups at init."
        )

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict):
        return self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}(opt={repr(self.optimizer)}, dht={repr(self.dht)})"

    def shutdown(self):
        raise NotImplementedError()
