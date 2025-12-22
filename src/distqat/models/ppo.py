import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from typing import Tuple

from hivemind.moe.server.layers.custom_experts import register_expert_class

import gymnasium as gym

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def head_sample_input(batch_size, in_dim):
    return (torch.empty((batch_size, in_dim)), torch.empty((batch_size, 1)))

@register_expert_class("ppo.full", head_sample_input)
class PPOAgent(nn.Module):
    def __init__(
        self,
        dataset_name: str | None = None,
        in_dim: int | None = None,
        action_dim: int | None = None,
        hid_dim: int = 64,
        **_kwargs,
    ):
        super().__init__()
        if in_dim is None or action_dim is None:
            if dataset_name is None:
                raise ValueError("PPOAgent requires either (in_dim, action_dim) or dataset_name to infer them")
            env = gym.make(dataset_name)
            try:
                obs_dim = int(np.array(env.observation_space.shape).prod())
                act_dim = int(np.array(env.action_space.shape).prod())
            finally:
                env.close()
            in_dim = obs_dim if in_dim is None else in_dim
            action_dim = act_dim if action_dim is None else action_dim

        self.in_dim = int(in_dim)
        self.action_dim = int(action_dim)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.in_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.in_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hid_dim, self.action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def forward(self, x, action=None):
        return self.get_action_and_value(x, action)

    def compute_gae_returns(
        self,
        *,
        next_obs: torch.Tensor,
        next_done: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns for PPO.

        Expects the same done convention used in `test_ppo.py`:
        - `dones[t]` is the done-flag for the *current* state s_t (stored before stepping).
        - `next_done` is the done-flag for the state after the last step (s_{T}).
        """
        num_steps = rewards.shape[0]
        device = rewards.device

        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        return advantages, returns


