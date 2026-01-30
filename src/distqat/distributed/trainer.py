import torch
# Avoid exhausting file descriptors under heavy tensor sharing (hivemind/torch mp reduction).
# "file_system" uses filesystem-based shared memory instead of per-storage fd passing.
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn.functional as F

import signal
import warnings
import time
from transformers import AutoConfig
import torch.nn as nn

import numpy as np

import click
import os
from pathlib import Path
from hivemind.utils.logging import get_logger
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import SchemaValidator

from distqat.distributed.model import SwarmBaselineModel
from distqat.distributed.model import SwarmModel, SwarmBaselineModel
from distqat.distributed.optim.diloco import TrainingProgressSchema
from distqat.config import Config, parse_args
from distqat.distributed.optim.diloco import TrainingState
from distqat.sharding import register_process, get_outer_step, set_outer_step_if_greater
from distqat.data import get_train_val_datasets, get_dataloader
from torch.utils.data import DataLoader
from distqat.data import collate_fn
from distqat.models.ppo import compute_gae_returns
from distqat.utils.loss import task_type_loss
logger = get_logger(__name__)
logger.setLevel('DEBUG')

class SwarmTrainer:
    """
    A trainer for distributed/collaborative training using hivemind swarm.
    
    This trainer can operate in two modes:
    1. Distributed mode: Uses remote experts for model computation via SwarmModel
    2. Baseline mode: Uses local model computation via SwarmBaselineModel
    
    Args:
        trainer_id: Unique identifier for this trainer instance
        config: Configuration containing data, model, network, and training parameters
        use_baseline_model: If True, uses local computation instead of remote experts
    """
    def __init__(self,
        trainer_id: int,
        config: Config,
        use_baseline_model: bool = False,
        *,
        dht=None,
        local_expert_backends=None,
    ):
        self.config = config
        self.trainer_id = trainer_id
        self.run_id = config.experiment_prefix + "_" + str(trainer_id)

        self.device = config.device
        
        self.num_warmup_steps = 0
        self.num_total_steps = config.diloco.total_steps

        self.use_baseline_model = use_baseline_model
        if not self.use_baseline_model:
            self.model = SwarmModel(
                config=self.config,
                trainer_id=self.trainer_id,
                dht=dht,
                local_expert_backends=local_expert_backends,
            )
        else:
            self.model = SwarmBaselineModel(
                config=self.config,
                trainer_id=self.trainer_id,
                disable_quant=config.disable_quant,
            )



        self.process_id = self.model.dht.peer_id.to_string()
        register_process(self.model.dht, self.config.experiment_prefix, ttl=300.0)
        # self.shm_client = DataClient(host=self.config.data_server.ipc_host, port=int(self.config.data_server.ipc_port), authkey=self.config.data_server.ipc_key.encode())

        self.batch_size = self.config.diloco.batch_size_per_step
        self.gradient_accumulation_steps = self.config.diloco.gradient_accumulation_steps
        self.inner_steps = self.config.diloco.inner_steps
        
        # if self.use_baseline_model and self.config.world_size > 0:
        #     self.batch_size *= self.config.world_size
        # if self.use_baseline_model:
        #     self.gradient_accumulation_steps *= self.config.world_size
            # self.inner_steps *= self.config.world_size
            
        effective_batch = self.batch_size * self.gradient_accumulation_steps
        
        logger.info(f"Trainer configured with:")
        logger.info(f"  - Micro-batch size: {self.batch_size}")
        logger.info(f"  - Accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  - Effective batch size: {effective_batch}")

        if self.config.data.task_type == "rl":
            self.dataloader = self._get_rl_dataloader()
        else:
            train_ds, _ = get_train_val_datasets(self.config.data)
            self.dataloader = get_dataloader(self.config, train_ds)
        
        self.remaining_batch = None
        self.remaining_release = None

    def parameters(self):
        yield from self.model.parameters()

    def _fetch_batch_with_retry(self):
        """
        Fetch a batch from the dataloader, handling worker crashes, dataset exhaustion, and missing files.
        Returns (uid, batch) tuple.
        """
        try:
            t0 = time.time()
            uid, batch = next(self.dataloader)
            dt = time.time() - t0
            logger.debug(f"[TRAINER:DataLoader] Fetch time: {dt:.4f}s")
            return uid, batch
        except StopIteration:
            logger.info("Dataset exhausted, recreating dataloader for next epoch")
            train_ds, _ = get_train_val_datasets(self.config.data)
            self.dataloader = get_dataloader(self.config, train_ds)
            t0 = time.time()
            uid, batch = next(self.dataloader)
            dt = time.time() - t0
            logger.debug(f"[TRAINER:DataLoader] Fetch time (reload): {dt:.4f}s")
            return uid, batch
        except (FileNotFoundError, RuntimeError) as e:
            error_str = str(e).lower()
            # Handle FileNotFoundError from missing parquet files in Hugging Face datasets
            if isinstance(e, FileNotFoundError) or "filenotfound" in error_str:
                logger.warning(f"FileNotFoundError in DataLoader (missing dataset file), recreating dataloader: {str(e)[:200]}")
                logger.warning("This can happen when a parquet file is temporarily unavailable. Recreating dataloader to get a new shard assignment.")
                time.sleep(1)  # Brief delay before retry
                train_ds, _ = get_train_val_datasets(self.config.data)
                self.dataloader = get_dataloader(self.config, train_ds)
                t0 = time.time()
                uid, batch = next(self.dataloader)
                dt = time.time() - t0
                logger.debug(f"[TRAINER:DataLoader] Fetch time (after FileNotFoundError): {dt:.4f}s")
                return uid, batch
            # Handle DataLoader worker crashes
            elif isinstance(e, RuntimeError) and "DataLoader worker" in str(e) and ("exited unexpectedly" in str(e) or "is killed" in str(e)):
                logger.warning(f"DataLoader worker crashed (likely OOM), recreating dataloader: {e}")
                logger.warning("Consider reducing num_workers in config if this happens frequently")
                train_ds, _ = get_train_val_datasets(self.config.data)
                self.dataloader = get_dataloader(self.config, train_ds)
                t0 = time.time()
                uid, batch = next(self.dataloader)
                dt = time.time() - t0
                logger.debug(f"[TRAINER:DataLoader] Fetch time (after worker crash): {dt:.4f}s")
                return uid, batch
            else:
                raise

    def _get_rl_dataloader(self):
        """
        Infinite iterator that yields PPO minibatches:
        inputs = (b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)
        labels = tensor([env_global_step], float32)  # for logging/monitoring
        """
        import gymnasium as gym
        from distqat.utils.buffer import RolloutBuffer

        cfg = self.config
        device = torch.device(cfg.device) if not isinstance(cfg.device, torch.device) else cfg.device

        # --- env factory (copied from old data_server.py; keep it local to avoid import side effects) ---
        def make_env(env_id, idx, gamma: float):
            def thunk():
                env = gym.make(env_id)
                env = gym.wrappers.FlattenObservation(env)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.ClipAction(env)
                env = gym.wrappers.NormalizeObservation(env)
                env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
                env = gym.wrappers.NormalizeReward(env, gamma=gamma)
                env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
                return env

            return thunk

        # --- PPO hyperparams ---
        num_envs = int(cfg.ppo.num_envs)
        num_steps = int(cfg.ppo.num_steps)
        update_epochs = int(cfg.ppo.update_epochs)
        minibatch_size = int(cfg.ppo.minibatch_size)
        gamma = float(cfg.ppo.gamma)
        gae_lambda = float(cfg.ppo.gae_lambda)

        if minibatch_size <= 0:
            raise ValueError(f"ppo.minibatch_size must be > 0, got {minibatch_size}")

        # Best-effort warning if the RL minibatch size doesn't match the trainer microbatch size.
        if minibatch_size != int(self.batch_size):
            logger.warning(
                f"RL minibatch_size (ppo.minibatch_size={minibatch_size}) != "
                f"trainer batch_size_per_step (diloco.batch_size_per_step={int(self.batch_size)}). "
                "This is allowed, but make sure it's intentional."
            )

        # --- build envs ---
        env_id = cfg.data.dataset_name
        envs = gym.vector.SyncVectorEnv([make_env(env_id, i, gamma) for i in range(num_envs)])
        try:
            assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

            obs_shape = tuple(envs.single_observation_space.shape)
            action_shape = tuple(envs.single_action_space.shape)

            # Prefer explicit dims from config, but fall back to env-derived values.
            in_dim_cfg = getattr(cfg.model_pipeline.pipeline[0], "in_dim", None)
            action_dim_cfg = getattr(cfg.model_pipeline.pipeline[0], "action_dim", None)
            in_dim = int(in_dim_cfg) if in_dim_cfg is not None else int(np.prod(obs_shape))
            action_dim = int(action_dim_cfg) if action_dim_cfg is not None else int(np.prod(action_shape))

            # Sanity check for FlattenObservation.
            if len(obs_shape) != 1 or obs_shape[0] != in_dim:
                logger.warning(
                    f"Env obs_shape={obs_shape} does not match in_dim={in_dim}. "
                    "If you recently changed wrappers or model dims, fix this mismatch."
                )

            rollout = RolloutBuffer(
                num_steps=num_steps,
                num_envs=num_envs,
                obs_shape=obs_shape,
                action_shape=action_shape,
                device=device,
                in_dim=in_dim,
                action_dim=action_dim,
            )

            # --- init episode state ---
            seed = (cfg.data.shuffle_seed or 42) + int(self.trainer_id) + 1
            np.random.seed(seed)
            torch.manual_seed(seed)

            next_obs_np, _ = envs.reset(seed=seed)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
            env_global_step = 0

            # PPO env metrics (episodic return/length) advance on an env-step clock, which is different
            # from the train-step clock used for losses/optimizer progress.
            # Publish these into a separate DHT key suffix so the monitor can aggregate them without
            # clobbering train-step metrics.
            from distqat.utils.metrics import MetricsLogger
            if not hasattr(self, "_rl_metrics_logger") or self._rl_metrics_logger is None:
                base = self.model.metrics_logger
                self._rl_metrics_logger = MetricsLogger(
                    dht=base.dht,
                    model=base.model,
                    local_public_key=base.local_public_key,
                    experiment_prefix=base.experiment_prefix,
                    statistics_expiration=base.statistics_expiration,
                    trainer_id=base.trainer_id,
                    key_suffix="_rl_metrics",
                )
            log_episodic = self._rl_metrics_logger.log_episodic_from_infos
            # log_episodic = self.model.metrics_logger.log_episodic_from_infos

            uid = 0
            batch_size = num_envs * num_steps
            if batch_size % minibatch_size != 0:
                raise ValueError(
                    f"ppo batch_size (num_envs*num_steps={batch_size}) must be divisible by "
                    f"ppo.minibatch_size ({minibatch_size})"
                )
            num_minibatches = batch_size // minibatch_size

            # Check if we should use averaged policy for rollouts (distributed RL)
            use_averaged_policy = getattr(cfg.ppo, 'use_averaged_policy_for_rollouts', False)
            if use_averaged_policy:
                logger.info("PPO configured to use averaged policy weights for rollout collection")

            while True:
                # Collect rollouts under current policy (this trainer's model).
                rollout.reset()
                agent = self.model
                
                rollout_start_time = time.time()
                
                # For remote models with averaged policy, we need to modify how rollouts are collected
                # The RolloutBuffer.collect_and_add_step uses agent(obs, action) internally
                # We'll create a wrapper that routes to forward_averaged when needed
                from contextlib import nullcontext
                
                if use_averaged_policy:
                    # Create a wrapper agent that uses forward_averaged for inference
                    class AveragedPolicyWrapper:
                        def __init__(self, model):
                            self._model = model
                        def __call__(self, inputs):
                            return self._model.forward_averaged(inputs)
                    rollout_agent = AveragedPolicyWrapper(self.model)
                    rollout_context = nullcontext()
                else:
                    rollout_agent = agent
                    rollout_context = nullcontext()
                
                with rollout_context:
                    for _ in range(0, num_steps):
                        env_global_step, next_obs, next_done = rollout.collect_and_add_step(
                            agent=rollout_agent,
                            envs=envs,
                            global_step=env_global_step,
                            next_obs=next_obs,
                            next_done=next_done,
                            log_episodic_from_infos=log_episodic,
                        )
                rollout_elapsed = time.time() - rollout_start_time
                logger.info(f"Rollout collection completed: {num_steps} steps in {rollout_elapsed:.2f}s ({rollout_elapsed/num_steps*1000:.1f}ms/step, {num_steps*num_envs/rollout_elapsed:.1f} env_steps/s)")

                # Compute GAE returns - also use averaged weights for next_value if applicable
                if use_averaged_policy:
                    gae_agent = rollout_agent
                else:
                    gae_agent = agent
                gae_context = nullcontext()
                
                with gae_context:
                    advantages, returns = compute_gae_returns(gae_agent, action_dim, next_obs, next_done, rollout.rewards, rollout.dones, rollout.values, gamma, gae_lambda)
                

                rollout.set_advantages_and_returns(advantages, returns)
                b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = rollout.get_batch()

                # Yield PPO minibatches for multiple epochs.
                for _epoch in range(update_epochs):
                    b_inds = torch.randperm(batch_size, device=b_obs.device)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]

                        mb_inputs = (
                            b_obs[mb_inds],
                            b_logprobs[mb_inds],
                            b_actions[mb_inds],
                            b_advantages[mb_inds],
                            b_returns[mb_inds],
                            b_values[mb_inds],
                        )
                        labels = torch.tensor([float(env_global_step)], device=b_obs.device, dtype=torch.float32)

                        uid += 1
                        yield uid, {"inputs": mb_inputs, "labels": labels}
        finally:
            try:
                envs.close()
            except Exception:
                pass



    def step(self, inner_step: int, step: int):
        raw_loss_sum = 0.0
        for accumulation_step in range(self.gradient_accumulation_steps):
            releases_to_call = []

            # For graph data, each batch is already complete and can't be sliced (because it's a tuple)
            # Skip the batching loop and use the batch directly
            if self.config.data.task_type == "node_pred" or self.config.data.task_type == "rl":
                uid, batch = self._fetch_batch_with_retry()
                inputs = batch["inputs"]
                labels = batch["labels"]
                if inner_step % 10 == 0 and accumulation_step == 0:
                    logger.info(f"Inner step {inner_step} of {self.inner_steps}")
            else:
                inputs_list = []
                labels_list = []
                
                samples_needed = self.batch_size
                
                while samples_needed > 0:
                    if self.remaining_batch is None:
                        uid, batch = self._fetch_batch_with_retry()
                        self.remaining_batch = batch
                    
                    current_inputs = self.remaining_batch["inputs"]
                    current_labels = self.remaining_batch["labels"]
                    available = current_inputs.shape[0]
                    
                    take = min(samples_needed, available)
                    
                    inputs_list.append(current_inputs[:take])
                    labels_list.append(current_labels[:take])
                    
                    samples_needed -= take
                    
                    if take == available:
                        releases_to_call.append(self.remaining_release)
                        self.remaining_batch = None
                        self.remaining_release = None
                    else:
                        self.remaining_batch["inputs"] = current_inputs[take:]
                        self.remaining_batch["labels"] = current_labels[take:]
                
                inputs = torch.cat(inputs_list)
                labels = torch.cat(labels_list)
                
                if inner_step % 10 == 0 and accumulation_step == 0:
                    logger.info(f"Inner step {inner_step} of {self.inner_steps}")

            
            if self.config.data.task_type == "image_gen":
                num_D_steps = self.config.biggan["num_D_steps"]

                outputs = self.model((inputs, labels))
                # BigGAN adapter serializes losses into an output tensor:
                # D_loss at index 0, G_loss at index -1
                D_loss_raw, G_loss_raw = outputs[0], outputs[-1]

                # Scale losses for gradient accumulation (backward uses scaled values).
                D_loss = D_loss_raw / self.gradient_accumulation_steps
                G_loss = G_loss_raw / self.gradient_accumulation_steps

                # Log individual losses as scalars so the monitor can forward them to wandb.
                # (We intentionally do not touch wandb directly from trainers.)
                try:
                    log_scalar = getattr(getattr(self.model, "metrics_logger", None), "log_scalar", None)
                    if callable(log_scalar):
                        log_scalar("losses/D_loss", float(D_loss_raw.detach().item()), step)
                        log_scalar("losses/G_loss", float(G_loss_raw.detach().item()), step)
                        if float(G_loss_raw.detach().item()) != 0.0:
                            log_scalar(
                                "losses/D_over_G",
                                float((D_loss_raw.detach() / G_loss_raw.detach()).item()),
                                step,
                            )
                except Exception:
                    pass
                
                # Log individual losses for GAN training diagnostics
                if inner_step % 10 == 0:
                    logger.info(f"Step {step}: D_loss={D_loss.item():.4f}, G_loss={G_loss.item():.4f}, D/G ratio={D_loss.item()/G_loss.item():.4f}, Sum={D_loss.item() + G_loss.item():.4f}")
                
                D_loss.backward(retain_graph=inner_step % num_D_steps == 0)
                if inner_step % num_D_steps == 0:
                    G_loss.backward()
                # Logging the sum for monitoring although it's not a meaningful metric
                loss = D_loss + G_loss
                # Unscale for logging (per-accumulation loss is scaled by 1/grad_accum)
                raw_loss_sum += float((loss.detach() * self.gradient_accumulation_steps).item())
            else:
                if self.config.data.task_type == "rl":
                    b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = inputs

                    outputs = self.model((b_obs, b_actions))
                elif self.config.data.task_type == "llm" or self.config.data.task_type == "node_pred":
                    outputs = self.model(inputs, labels)
                else:
                    outputs = self.model(inputs)

                
                raw_loss = task_type_loss(self.config, inputs, outputs, labels, self.model.metrics_logger, step)
                raw_loss_sum += float(raw_loss.detach().mean().item())
                loss = raw_loss / self.gradient_accumulation_steps
                if not torch.isfinite(loss).all():
                    logger.warning(f"Non-finite loss at step={step} (inner_step={inner_step}): {loss.item()}. Skipping update.")
                    self.model.zero_grad(set_to_none=True)
                    return
                loss.backward()
            

            for release in releases_to_call:
                try:
                    release()
                except Exception:
                    pass

        # Log a stable, comparable value: mean raw loss over the accumulation window.
        mean_raw_loss = raw_loss_sum / float(self.gradient_accumulation_steps)

        if self.use_baseline_model and self.config.diloco.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.diloco.max_grad_norm))

        self.model.post_optimizer_callback(step, mean_raw_loss)

    def train(self):
        logger.info(f"============= Training for {self.num_total_steps} steps =============")

        inner_steps = self.inner_steps
        outer_step = 0
        prefix = self.config.experiment_prefix
        dht = self.model.dht 
        signature_validator = RSASignatureValidator()
        dht.add_validators([SchemaValidator(TrainingProgressSchema, prefix=prefix), signature_validator])
    
        step = 0

        logger.info(f"============= Checking Progress of other Servers =============")
        while True:
            max_outer_step_progress_ratio = 0.0
            n_stages = len(self.config.model_pipeline.pipeline)
            for stage_index in range(n_stages):
                summed_inner_steps, n_servers = 0, 0

                progress_dict = dht.get(f"{prefix}_{stage_index}_progress", latest=True)
                if progress_dict is not None:
                    progress_dict = progress_dict.value
                    progress = [
                        TrainingState.validate(entry.value)
                        for entry in progress_dict.values()
                        if entry.value is not None
                    ]
                    summed_inner_steps += sum(p.inner_step for p in progress)
                    n_servers = len(progress)
                    if n_servers > 0 and inner_steps > 0:
                        outer_step_progress_ratio = summed_inner_steps / (n_servers * inner_steps)
                    else:
                        outer_step_progress_ratio = 0.0
                else:
                    outer_step_progress_ratio = 0.0

                max_outer_step_progress_ratio = max(
                    max_outer_step_progress_ratio,
                    outer_step_progress_ratio,
                )

            if max_outer_step_progress_ratio < 0.2:
                break

            logger.info(f"Waiting for other servers to progress... (ratio: {max_outer_step_progress_ratio})")
            time.sleep(1.0)

        logger.info(f"============= Starting training =============")

        # TODO: Flatten the loops back to while not done and add the rebuild_pipeline step with a modulo step based on a config parameter
        while step < self.num_total_steps:
            logger.info(f"Outer step {outer_step} starting")

            # heartbeat into active-set with TTL if you have a refresher
            register_process(dht, prefix, ttl=300.0)
            
            # (Re)build sharded pipeline for this outer step 
            shared = get_outer_step(dht, prefix)
            outer_step = max(outer_step, shared)
            
            # Inner loop: exactly inner_steps batches for this peer
            inner = 0

            while inner < inner_steps:
                self.step(inner_step=inner, step=step)
                
                # Only increment global step (optimizer step) when we actually stepped
                # if (inner + 1) % self.gradient_accumulation_steps == 0:
                step += 1
                inner += 1
                if step >= self.num_total_steps:
                    break
            
            logger.info(f"Outer step {outer_step} completed")
            set_outer_step_if_greater(dht, prefix, outer_step + 1)

            outer_step += 1

        logger.info(f"============= Training finished =============")

    def shutdown(self):
        try:
            if hasattr(self, "_rl_metrics_logger") and self._rl_metrics_logger is not None:
                self._rl_metrics_logger.shutdown()
        except Exception:
            pass
        self.model.shutdown()


def main(cfg: Config, trainer_id: int, run_locally: bool):
    trainer = SwarmTrainer(
        trainer_id=trainer_id,
        config=cfg,
        use_baseline_model=run_locally,
    )
    signal.signal(signal.SIGINT, signal.default_int_handler)
    
    logger.info(f"Created SwarmTrainer with config from {cfg}")
    logger.info(f"Trainer ID: {trainer_id}")
    logger.info(f"Using baseline model: {run_locally}")

    try:    
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error in trainer: {e}")
        raise e
    finally:
        trainer.shutdown()

if __name__ == "__main__":
    parse_args_with_extra_kwargs = click.option("--trainer-id", type=int)(parse_args)
    parse_args_with_extra_kwargs = click.option("--run-locally", is_flag=True)(parse_args_with_extra_kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*is used more than once. Remove its duplicate as parameters should be unique.*")
        res = parse_args_with_extra_kwargs(standalone_mode=False)
        if isinstance(res, int):
            quit() # Help has been called
        elif isinstance(res, tuple):
            cfg, extra_kwargs = res
            main(cfg, **extra_kwargs)
        else:
            raise ValueError(f"Unexpected return type: {type(res)}")
