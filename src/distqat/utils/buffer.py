import torch
import numpy as np
from typing import Tuple, Callable
import torch.nn as nn

class RolloutBuffer:
    """
    Fixed-size rollout storage for PPO.

    **Done convention (matches the existing `test_ppo.py` logic)**:
    - We store `dones[t] = done(s_t)` (the done flag for the *current* state).
    - After stepping, we compute `next_done = done(s_{t+1})` and carry it into the next step.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        in_dim: int,
        action_dim: int,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.in_dim = in_dim
        self.action_dim = action_dim

        self.step_idx = 0

    def reset(self) -> None:
        self.step_idx = 0

    def collect_and_add_step(
        self,
        *,
        agent: nn.Module,
        envs,
        global_step: int,
        next_obs: torch.Tensor,
        next_done: torch.Tensor,
        log_episodic_from_infos: Callable[[dict, int], None],
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Collect one environment step and write it into storage at `self.step_idx`.
        Returns updated (global_step, next_obs, next_done).
        """
        if self.step_idx >= self.num_steps:
            raise IndexError("RolloutBuffer is full; call reset() before collecting again.")

        step = self.step_idx
        global_step += self.num_envs

        # Store current state + done flag (done(s_t))
        self.obs[step] = next_obs
        self.dones[step] = next_done

        # Action/value under current policy
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            self.values[step] = value.flatten()
        self.actions[step] = action
        self.logprobs[step] = logprob

        # Step env (produces s_{t+1}, r_t, done(s_{t+1}))
        next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done_np = np.logical_or(terminations, truncations)

        self.rewards[step] = torch.as_tensor(reward, device=self.device, dtype=torch.float32).view(-1)

        next_obs = torch.as_tensor(next_obs_np, device=self.device, dtype=torch.float32)
        next_done = torch.as_tensor(next_done_np, device=self.device, dtype=torch.float32)

        log_episodic_from_infos(infos, global_step)

        self.step_idx += 1
        return global_step, next_obs, next_done

    def set_advantages_and_returns(self, advantages: torch.Tensor, returns: torch.Tensor) -> None:
        self.advantages = advantages
        self.returns = returns

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # flatten the batch
        b_obs = self.obs.reshape((-1, self.in_dim))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1, self.action_dim))
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values
