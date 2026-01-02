import torch
torch.multiprocessing.set_sharing_strategy('file_descriptor')
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
from distqat.models.wav2vec2 import get_feat_extract_output_lengths
from distqat.distributed.data_client import DataClient
from distqat.data import get_train_val_datasets
from torch.utils.data import DataLoader
from distqat.distributed.data_server import BufferedShuffleIterable
from distqat.data import collate_fn

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
        disable_quant: bool = False,
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
            )
        else:
            self.model = SwarmBaselineModel(
                config=self.config,
                trainer_id=self.trainer_id,
                disable_quant=disable_quant,
            )

        # Mixed precision stability: GradScaler prevents overflow from corrupting weights.
        self.scaler = None
        # if (
        #     self.use_baseline_model
        #     and self.config.data.precision == "fp16-mixed"
        #     and isinstance(self.device, str)
        #     and self.device.startswith("cuda")
        # ):
        #         self.scaler = torch.amp.GradScaler()
    

        self.process_id = self.model.dht.peer_id.to_string()
        register_process(self.model.dht, self.config.experiment_prefix, ttl=300.0)
        # self.shm_client = DataClient(host=self.config.data_server.ipc_host, port=int(self.config.data_server.ipc_port), authkey=self.config.data_server.ipc_key.encode())

        self.batch_size = self.config.diloco.batch_size_per_step
        self.gradient_accumulation_steps = self.config.diloco.gradient_accumulation_steps
        self.inner_steps = self.config.diloco.inner_steps
        
        # if self.use_baseline_model and self.config.world_size > 0:
        #     self.batch_size *= self.config.world_size
        if self.use_baseline_model:
            self.gradient_accumulation_steps *= self.config.world_size
            # self.inner_steps *= self.config.world_size
            
        effective_batch = self.batch_size * self.gradient_accumulation_steps
        
        logger.info(f"Trainer configured with:")
        logger.info(f"  - Micro-batch size: {self.batch_size}")
        logger.info(f"  - Accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"  - Effective batch size: {effective_batch}")

        self.dataloader = self.get_dataloader()
        
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
            self.dataloader = self.get_dataloader()
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
                self.dataloader = self.get_dataloader()
                t0 = time.time()
                uid, batch = next(self.dataloader)
                dt = time.time() - t0
                logger.debug(f"[TRAINER:DataLoader] Fetch time (after FileNotFoundError): {dt:.4f}s")
                return uid, batch
            # Handle DataLoader worker crashes
            elif isinstance(e, RuntimeError) and "DataLoader worker" in str(e) and ("exited unexpectedly" in str(e) or "is killed" in str(e)):
                logger.warning(f"DataLoader worker crashed (likely OOM), recreating dataloader: {e}")
                logger.warning("Consider reducing num_workers in config if this happens frequently")
                self.dataloader = self.get_dataloader()
                t0 = time.time()
                uid, batch = next(self.dataloader)
                dt = time.time() - t0
                logger.debug(f"[TRAINER:DataLoader] Fetch time (after worker crash): {dt:.4f}s")
                return uid, batch
            else:
                raise

    def get_dataloader(self):
        ds, _ = get_train_val_datasets(self.config.data)
        # ds = BufferedShuffleIterable(ds, buffer_size=self.config.data.shuffle_buffer_size, seed=0)
        loader = DataLoader(
            ds,                                 # yields (uid, sample) or sample
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=collate_fn(self.config.data, self.config.model_pipeline.pipeline[0]),
            persistent_workers=self.config.data.num_workers > 0,
            prefetch_factor=2 if self.config.data.num_workers > 0 else None,
        )
        return iter(loader)

    def task_type_loss(self, inputs, outputs, labels, step=None):
        if self.config.data.task_type == "cv":
            return F.cross_entropy(outputs.float(), labels.to(outputs.device))
        elif self.config.data.task_type == "llm":
            # shift for next-token prediction
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.float().permute(0, 2, 1),
                shift_labels.to(shift_logits.device),
                reduction="none",
            )
            loss = loss.mean()
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Found {loss.item()} loss in llm task!")
            return loss
        elif self.config.data.task_type == "speech":
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
            model_name = self.config.data.full_model_name
            input_lengths = get_feat_extract_output_lengths(attention_mask.sum(-1), config=AutoConfig.from_pretrained(model_name)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = F.log_softmax(outputs, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    reduction="mean",
                )
            return loss
        elif self.config.data.task_type == "image_gen":
            D_loss, G_loss = outputs['D_loss'], outputs['G_loss']
            D_loss.backward()
            G_loss.backward()
        elif self.config.data.task_type == "node_pred":
            outputs = outputs.squeeze(0)
            labels = labels.squeeze(0)
            return F.cross_entropy(outputs, labels)
        elif self.config.data.task_type == "rl":
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = inputs
            _, newlogprob, entropy, newvalue = outputs
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_coef = float(self.config.ppo.clip_coef)
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

            mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - b_returns) ** 2
            v_clipped = b_values + torch.clamp(newvalue - b_values, -clip_coef, clip_coef)
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - float(self.config.ppo.ent_coef) * entropy_loss + v_loss * float(self.config.ppo.vf_coef)

            # logging
            y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            log_scalar = self.model.metrics_logger.log_scalar
            log_scalar("losses/value_loss", v_loss.item(), step)
            log_scalar("losses/policy_loss", pg_loss.item(), step)
            log_scalar("losses/entropy", entropy_loss.item(), step)
            log_scalar("losses/old_approx_kl", old_approx_kl.item(), step)
            log_scalar("losses/approx_kl", approx_kl.item(), step)
            log_scalar("losses/clipfrac", clipfrac, step)
            log_scalar("losses/explained_variance", float(explained_var), step)
            log_scalar("charts/global_step", step, step)

            return loss
        else:
            raise ValueError(f"Unknown task type: {self.config.data.task_type}")


    def step(self, inner_step: int, step: int):
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
                D_loss, G_loss = outputs[0], outputs[-1]
                D_loss = D_loss / self.gradient_accumulation_steps
                G_loss = G_loss / self.gradient_accumulation_steps
                
                # Log individual losses for GAN training diagnostics
                if inner_step % 10 == 0:
                    logger.info(f"Step {step}: D_loss={D_loss.item():.4f}, G_loss={G_loss.item():.4f}, D/G ratio={D_loss.item()/G_loss.item():.4f}, Sum={D_loss.item() + G_loss.item():.4f}")
                
                D_loss.backward(retain_graph=inner_step % num_D_steps == 0)
                if inner_step % num_D_steps == 0:
                    G_loss.backward()
                # Logging the sum for monitoring although it's not a meaningful metric
                loss = D_loss + G_loss
            else:
                if self.config.data.task_type == "rl":
                    b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = inputs
                    outputs = self.model((b_obs, b_actions))
                else:
                    outputs = self.model(inputs)
                loss = self.task_type_loss(inputs, outputs, labels, step)
                loss = loss / self.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss at step={step} (inner_step={inner_step}): {loss.item()}. Skipping update.")
                    self.model.zero_grad(set_to_none=True)
                    return
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            

            # Only step optimizer if we have accumulated enough gradients
            if self.use_baseline_model:
                # For baseline model the optimzer callback steps the optimizer so only step if we have accumulated enough gradients
                if (inner_step + 1) % self.gradient_accumulation_steps == 0:
                    if self.config.diloco.max_grad_norm is not None:
                        # TODO: scaling needs to be implemented for server models as well, otherwise comparison with baseline model is not fair
                        if self.scaler is not None and hasattr(self.model, "optimizer"):
                            # clip on unscaled gradients under AMP
                            self.scaler.unscale_(self.model.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.diloco.max_grad_norm))
                    self.model.post_optimizer_callback(
                        step,
                        loss.item() * self.gradient_accumulation_steps,
                        scaler=self.scaler,
                    )
            else:
                # For swarm model the optimzer callback does not step the optimizer so log the loss every inner step
                self.model.post_optimizer_callback(step, loss.item() * self.gradient_accumulation_steps)

            if self.config.data.task_type == "image_gen":
                if inner_step % self.config.biggan["eval_every"] == 0:
                    self.model.evaluate(step)
            
            for release in releases_to_call:
                try:
                    release()
                except Exception:
                    pass

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
        self.model.shutdown()


def main(cfg: Config, trainer_id: int, run_locally: bool, disable_quant: bool = False):
    trainer = SwarmTrainer(
        trainer_id=trainer_id,
        config=cfg,
        use_baseline_model=run_locally,
        disable_quant=disable_quant,
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
    parse_args_with_extra_kwargs = click.option("--disable-quant", is_flag=True)(parse_args_with_extra_kwargs)

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
