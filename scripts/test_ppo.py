# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Callable, Tuple
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from pydantic_yaml import parse_yaml_file_as

from distqat.config import Config

from distqat.models.ppo import PPOAgent
from distqat.optimizers import get_optimizer_factory

from distqat.utils.buffer import RolloutBuffer

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO (CleanRL) using distqat YAML config for experiment metadata/runtime."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/ppo.yaml",
        help="Path to a distqat YAML config (e.g. configs/ppo.yaml).",
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        default=1,
        help="Number of servers to use.",
    )
    return parser.parse_args()


def _load_cfg(config_path: str) -> Config:
    cfg = parse_yaml_file_as(Config, config_path)
    cfg.path = config_path
    return cfg


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # Gymnasium requires an explicit observation_space for TransformObservation in newer versions.
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk




if __name__ == "__main__":
    cli = _parse_cli()
    cfg = _load_cfg(cli.config_path)

    # Map distqat config -> PPO script settings
    env_id = cfg.data.dataset_name
    exp_name = cfg.experiment_prefix or os.path.basename(__file__)[: -len(".py")]
    seed = 42

    # Prefer cfg.device ("cuda"/"cpu"/"rocm"), but always fall back safely.
    want_cuda = str(cfg.device).lower() == "cuda"
    device = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")

    # Keep the CleanRL run name format, but store logs under cfg.log_dir.
    run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"
    hparams = {
        "config_path": cli.config_path,
        "env_id": env_id,
        "device": str(device),
        "log_dir": str(cfg.log_dir),
        "wandb_project": cfg.wandb_project,
        "wandb_entity": cfg.wandb_entity,
        **cfg.ppo.model_dump(),
    }

    # Optional W&B: enabled iff wandb_project is set in YAML
    wandb_run = None
    if cfg.wandb_project is not None:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=hparams,
                name=run_name,
                save_code=True,
            )
        except Exception as e:
            print(f"[WARN] W&B enabled by config but failed to init: {e}")

    def log_scalar(name: str, value, step: int) -> None:
        if wandb_run is not None:
            wandb_run.log({name: value}, step=step)

    def log_episodic_from_infos(infos: dict, step: int) -> None:
        """
        Gymnasium vector envs have had multiple info formats over time.
        We support:
        - infos["final_info"] = list[dict] where dict may contain {"episode": {...}}
        - infos["episode"] = {"r": np.ndarray|float, "l": np.ndarray|int, ...} (or list of dicts)
        """
        if not isinstance(infos, dict):
            return


        ep = infos.get("episode")
        if ep is None:
            return
        if isinstance(ep, dict):
            r = ep.get("r")
            l = ep.get("l")
            # r/l may be scalar or per-env arrays/lists
            if isinstance(r, (list, tuple, np.ndarray)):
                for i, ri in enumerate(r):
                    if ri is None:
                        continue
                    li = None
                    if isinstance(l, (list, tuple, np.ndarray)) and i < len(l):
                        li = l[i]
                    print(f"global_step={step}, episodic_return={ri}")
                    log_scalar("charts/episodic_return", ri, step)
                    if li is not None:
                        log_scalar("charts/episodic_length", li, step)
            else:
                print(f"global_step={step}, episodic_return={r}")
                log_scalar("charts/episodic_return", r, step)
                if l is not None:
                    log_scalar("charts/episodic_length", l, step)

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id,
                i,
                False,  # capture_video removed (not in distqat config)
                run_name,
                float(cfg.ppo.gamma),
            )
            for i in range(int(cfg.ppo.num_envs))
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = PPOAgent(dataset_name=cfg.data.dataset_name, in_dim=cfg.model_pipeline.pipeline[0].in_dim, action_dim=cfg.model_pipeline.pipeline[0].action_dim).to(device)
    # optimizer = optim.Adam(agent.parameters(), lr=float(PPO_DEFAULTS["learning_rate"]), eps=1e-5)
    optimizer = get_optimizer_factory(cfg.diloco.inner_optim)(agent.parameters())

    # ALGO Logic: Storage setup
    num_steps = int(cfg.ppo.num_steps) # default:2048
    update_epochs = int(cfg.ppo.update_epochs) # default: 10
    # num_envs = int(cli.num_servers) # default:1
    num_envs = int(cfg.ppo.num_envs)
    batch_size = int(num_envs * num_steps) # default:2048
    minibatch_size = int(cfg.ppo.minibatch_size) # num_minibatches = batch_size_per_step, default: 32; num_minibatches = batch_size / batch_size_per_step = 64
    num_iterations = int(cfg.ppo.total_timesteps / batch_size)

    rollout = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_shape=tuple(envs.single_observation_space.shape),
        action_shape=tuple(envs.single_action_space.shape),
        device=device,
        in_dim=cfg.model_pipeline.pipeline[0].in_dim,
        action_dim=cfg.model_pipeline.pipeline[0].action_dim,
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(num_envs).to(device)

    for iteration in range(1, num_iterations + 1):
        rollout.reset()

        # Collect data from the environment
        # import cProfile
        # import pstats
        # import io

        # pr = cProfile.Profile()
        # pr.enable()

        for _ in range(0, num_steps):
            global_step, next_obs, next_done = rollout.collect_and_add_step(
                agent=agent,
                envs=envs,
                global_step=global_step,
                next_obs=next_obs,
                next_done=next_done,
                log_episodic_from_infos=log_episodic_from_infos,
            )

        # pr.disable()
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        # # ps.print_stats(20)  # print top 20 functions
        # print("---- cProfile for rollout.collect_step loop ----")
        # print(s.getvalue())

        advantages, returns = agent.compute_gae_returns(
            next_obs=next_obs,
            next_done=next_done,
            rewards=rollout.rewards,
            dones=rollout.dones,
            values=rollout.values,
            gamma=float(cfg.ppo.gamma),
            gae_lambda=float(cfg.ppo.gae_lambda),
        )
        rollout.set_advantages_and_returns(advantages, returns)

        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = rollout.get_batch()

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []

        # Actually maybe I can use this as my datasharding method. Each server/ worker just gets part of the rollout buffer data. 
        # Maybe instead of update_epochs, I can use num_servers. And then on each server the for loop will just go over the batch_size and not be multiplied by update_epochs.
        # This way we can store them in the dataserver with batch size and each trainer takes a part of the batch.
        # mega_b_inds = torch.cat([torch.randperm(batch_size) for _ in range(update_epochs)])

        # pr_inner = cProfile.Profile()
        # pr_inner.enable()

        # for start in range(0, batch_size * update_epochs, minibatch_size):
        for epoch in range(update_epochs):
            b_inds = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                # mb_inds = mega_b_inds[start:end]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_coef = float(cfg.ppo.clip_coef)
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                clip_coef = float(cfg.ppo.clip_coef)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                clip_coef = float(cfg.ppo.clip_coef)
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - float(cfg.ppo.ent_coef) * entropy_loss + v_loss * float(cfg.ppo.vf_coef)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), float(cfg.diloco.max_grad_norm))
                optimizer.step()

                if cfg.ppo.target_kl is not None and approx_kl > float(cfg.ppo.target_kl):
                    break

        # pr_inner.disable()
        # s_inner = io.StringIO()
        # ps_inner = pstats.Stats(pr_inner, stream=s_inner).sort_stats('cumulative')
        # # ps_inner.print_stats(20)  # print top 20 functions for the inner loop
        # print("---- cProfile for PPO update loop ----")
        # print(s_inner.getvalue())

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        log_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        log_scalar("losses/value_loss", v_loss.item(), global_step)
        log_scalar("losses/policy_loss", pg_loss.item(), global_step)
        log_scalar("losses/entropy", entropy_loss.item(), global_step)
        log_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        log_scalar("losses/approx_kl", approx_kl.item(), global_step)
        log_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        log_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        log_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        log_scalar("charts/global_step", global_step, global_step)


    envs.close()
    if wandb_run is not None:
        wandb_run.finish()