#!/usr/bin/env python3
"""
Single-shot evaluation script for GCN workers.
Mirrors parameters from workers via DiLoCo optimizer and runs one evaluation on
the validation split. Logs metrics to wandb and JSON.
Joins the same wandb run as monitor/workers via DHT-shared run ID.

Periodic execution is handled by systemd timer (see start_servers.yaml).
"""

import argparse
import gc
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
# Use file_system sharing strategy to reduce shared memory issues.
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
from hivemind.dht import DHT
from hivemind.moe.server.layers import name_to_block
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from pydantic_yaml import parse_yaml_file_as
from torch.utils.data import DataLoader

from distqat.config import Config
from distqat.data import collate_fn, get_train_val_datasets
from distqat.models import kwargs_from_config
from distqat.optimizers import get_diloco_optimizer_cls_kwargs
from distqat.utils.compression import get_compression_kwargs
from distqat.utils.logging import get_wandb_run_id_with_retries
from distqat.utils.loss import task_type_loss

logger = get_logger(__name__)
use_hivemind_log_handler("in_root_logger")

# Configuration
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "gcn.yaml"
INITIAL_PEERS_PATH = Path(__file__).resolve().parents[3] / "logs" / "gcn" / "initial_peers.txt"
EVAL_RESULTS_DIR = Path(__file__).resolve().parents[3] / "logs" / "gcn" / "eval_results"
MAX_BATCHES = 50
SPLIT = "valid"


class EvalParamMirror:
    """
    Mirrors parameters from workers via DiLoCo optimizer.
    Similar to ParamMirror but designed for evaluation use case.
    """

    # Maximum consecutive refresh failures before requiring restart
    MAX_CONSECUTIVE_FAILURES = 5
    # Shorter timeout for evaluation (30 seconds instead of 10 minutes)
    LOAD_STATE_TIMEOUT = 30.0

    def __init__(self, cfg: Config, dht: DHT):
        self.cfg = cfg
        self.dht = dht
        self._mirrors: List[Tuple[torch.nn.Module, object, str, int]] = []  # model, optimizer, run_id, stage_index
        self._compression = get_compression_kwargs(cfg.network.hivemind_compression)
        self._consecutive_failures = 0
        self._needs_optimizer_recreation = False

        models: List[str] = []
        stages: List[str] = []
        for pipeline_step_cfg in cfg.model_pipeline.pipeline:
            model, stage = pipeline_step_cfg.model_name.split(".")
            models.append(model)
            stages.append(stage)

        for stage_index, pipeline_step_cfg in enumerate(cfg.model_pipeline.pipeline):
            expert_cls = f"{models[stage_index]}.{stages[stage_index]}"
            block_ctor = name_to_block.get(expert_cls)
            if block_ctor is None:
                logger.warning(f"EvalParamMirror: unknown expert_cls {expert_cls}, skipping stage {stage_index}")
                continue

            model_kwargs = kwargs_from_config(block_ctor.__init__, pipeline_step_cfg, cfg.data)
            model = block_ctor(**model_kwargs)
            model.to("cpu")
            model.eval()

            run_id = f"{cfg.experiment_prefix}_{stage_index}"
            optimizer = self._create_optimizer(model, run_id)
            if optimizer is None:
                continue

            # Initial parameter sync from peers - use direct averager call to avoid infinite retry loop
            try:
                self._load_state_with_timeout(optimizer)
                # Clone parameters to avoid shared memory issues when subprocess exits
                self._clone_model_parameters(model)
                logger.info(f"Loaded initial state from peers for stage {stage_index}")
            except Exception as e:
                logger.warning(f"EvalParamMirror: failed to load state from peers for stage {stage_index}: {e}")

            self._mirrors.append((model, optimizer, run_id, stage_index))

        # Final cleanup to release any shared memory from subprocess
        gc.collect()
        logger.info(f"EvalParamMirror initialized with {len(self._mirrors)} stage(s)")

    def _create_optimizer(self, model: torch.nn.Module, run_id: str) -> Optional[object]:
        """Create a DiLoCo optimizer for the given model with shorter timeout."""
        optim_cls, optim_kwargs = get_diloco_optimizer_cls_kwargs(run_id, self.cfg.diloco, self._compression)
        # Override timeout to be shorter for evaluation
        optim_kwargs["load_state_timeout"] = self.LOAD_STATE_TIMEOUT
        try:
            return optim_cls(
                params=model.parameters(),
                avg_only_params=[],
                dht=self.dht,
                **optim_kwargs,
            )
        except Exception as e:
            logger.warning(f"EvalParamMirror: failed to create optimizer for {run_id}: {e}")
            return None

    def _load_state_with_timeout(self, optimizer: object) -> None:
        """
        Load state from peers with controlled timeout.
        
        This calls the averager directly instead of optimizer.load_state_from_peers()
        because the optimizer's method has an infinite retry loop that would hang forever
        if there's an issue with shared memory cleanup.
        """
        # Access the underlying averager to avoid the optimizer's infinite retry loop
        if hasattr(optimizer, 'averager'):
            optimizer.averager.load_state_from_peers(timeout=self.LOAD_STATE_TIMEOUT)
        else:
            # Fallback to direct call if structure is different
            optimizer.load_state_from_peers()

    def _clone_model_parameters(self, model: torch.nn.Module) -> None:
        """
        Clone all model parameters in-place to avoid shared memory backing.
        This prevents 'could not unlink shared memory file' errors when
        subprocesses (from load_state_from_peers) exit.
        """
        with torch.no_grad():
            for param in model.parameters():
                # Clone and detach to ensure complete separation from shared memory
                param.data = param.data.clone().detach()
            for buffer in model.buffers():
                buffer.data = buffer.data.clone().detach()
        # Force garbage collection to release shared memory references
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def refresh_parameters(self) -> bool:
        """
        Sync parameters from workers. Returns True if at least one stage was synced.
        If a TimeoutError occurs (usually after subprocess crash), marks optimizer
        for recreation on next attempt rather than blocking.
        
        Raises RuntimeError if MAX_CONSECUTIVE_FAILURES is exceeded.
        """
        # If we previously had a failure, recreate optimizers first
        if self._needs_optimizer_recreation:
            logger.info("Recreating optimizers after previous failure...")
            self._recreate_all_optimizers()
            self._needs_optimizer_recreation = False
        
        success = False
        for i, (model, optimizer, run_id, stage_index) in enumerate(self._mirrors):
            try:
                logger.debug(f"Loading state from peers for {run_id}...")
                # Use direct averager call to avoid optimizer's infinite retry loop
                self._load_state_with_timeout(optimizer)
                # Clone parameters to avoid shared memory issues when subprocess exits
                self._clone_model_parameters(model)
                success = True
                logger.debug(f"Refreshed parameters for {run_id}")
            except (TimeoutError, Exception) as e:
                is_timeout = isinstance(e, TimeoutError) or "TimeoutError" in type(e).__name__
                if is_timeout:
                    # Subprocess likely crashed - don't block trying to recover now
                    # Mark for recreation and continue with existing parameters
                    logger.warning(
                        f"TimeoutError for {run_id} - subprocess may have crashed. "
                        f"Will recreate optimizer on next refresh attempt."
                    )
                    self._needs_optimizer_recreation = True
                else:
                    logger.warning(f"Failed to refresh parameters for {run_id}: {e}")
        
        # Track consecutive failures
        if not success:
            self._consecutive_failures += 1
            logger.warning(
                f"Parameter refresh failed ({self._consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES} consecutive failures)"
            )
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Parameter refresh failed {self.MAX_CONSECUTIVE_FAILURES} times consecutively. "
                    f"Evaluation service needs restart."
                )
        else:
            if self._consecutive_failures > 0:
                logger.info(f"Parameter refresh recovered after {self._consecutive_failures} failure(s)")
            self._consecutive_failures = 0
        
        # Additional cleanup after all refreshes
        gc.collect()
        return success
    
    def _recreate_all_optimizers(self) -> None:
        """Recreate all optimizers after a failure."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        new_mirrors = []
        for model, old_optimizer, run_id, stage_index in self._mirrors:
            try:
                # Shutdown old optimizer if possible
                if hasattr(old_optimizer, 'shutdown'):
                    try:
                        old_optimizer.shutdown()
                    except Exception:
                        pass
            except Exception:
                pass
            
            new_optimizer = self._create_optimizer(model, run_id)
            if new_optimizer is not None:
                new_mirrors.append((model, new_optimizer, run_id, stage_index))
                logger.info(f"Recreated optimizer for {run_id}")
            else:
                # Keep old optimizer as fallback
                new_mirrors.append((model, old_optimizer, run_id, stage_index))
                logger.warning(f"Failed to recreate optimizer for {run_id}, keeping old one")
        
        self._mirrors = new_mirrors
        gc.collect()

    def get_model(self) -> Optional[torch.nn.Module]:
        """
        Returns the first stage model for evaluation.
        """
        if not self._mirrors:
            return None
        return self._mirrors[0][0]

    def get_all_models(self) -> List[torch.nn.Module]:
        """Returns all stage models."""
        return [model for model, _, _, _ in self._mirrors]


def _get_initial_peers(cfg: Config) -> list[str]:
    """Get initial peers from file or config."""
    if INITIAL_PEERS_PATH.exists():
        file_content = INITIAL_PEERS_PATH.read_text().strip()
        peers = [p.strip() for p in file_content.split(",") if p.strip()]
        if peers:
            return peers

    if cfg.network.initial_peers:
        return cfg.network.initial_peers

    return []


def _init_dht(cfg: Config) -> Optional[DHT]:
    """Initialize DHT connection."""
    initial_peers = _get_initial_peers(cfg)
    if not initial_peers:
        logger.error("No initial peers found, cannot connect to DHT")
        return None

    try:
        dht = DHT(start=True, initial_peers=initial_peers)
        logger.info(f"Connected to DHT with {len(initial_peers)} initial peer(s)")
        return dht
    except Exception as e:
        logger.error(f"Failed to connect to DHT: {e}")
        return None


def _init_wandb(cfg: Config, dht: DHT, wandb_run_id_path: Optional[str] = None) -> bool:
    """Initialize wandb for logging evaluation metrics. Returns True if successful.

    Reads the wandb run ID from file if provided, otherwise retrieves from DHT.
    """
    if cfg.wandb_project is None:
        logger.info("Wandb project not configured, skipping wandb logging")
        return False

    wandb_run_id = None
    
    # Try reading from file first (more reliable than DHT)
    if wandb_run_id_path:
        try:
            wandb_run_id_file = Path(wandb_run_id_path)
            if wandb_run_id_file.exists():
                wandb_run_id = wandb_run_id_file.read_text().strip()
                if wandb_run_id:
                    logger.info(f"Read wandb_run_id from file: {wandb_run_id}")
        except Exception as e:
            logger.warning(f"Failed to read wandb_run_id from file: {e}")
    
    # Fall back to DHT if file not available
    if not wandb_run_id:
        wandb_run_id = get_wandb_run_id_with_retries(
            dht, cfg.experiment_prefix, max_retries=10, retry_delay=2.0
        )
        if wandb_run_id:
            logger.info(f"Retrieved wandb_run_id from DHT: {wandb_run_id}")
    
    if not wandb_run_id:
        logger.warning("wandb_run_id not found in file or DHT, evaluation metrics won't be logged")
        return False

    try:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"{cfg.experiment_prefix}",
            id=wandb_run_id,
            resume="allow",
        )
        # Define evaluation-specific metrics
        wandb.define_metric("eval/*", step_metric="eval/timestamp")
        logger.info(f"Wandb initialized with shared run: {cfg.wandb_project}/{wandb_run_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        return False


def _log_to_wandb(metrics: Dict[str, Any], wandb_enabled: bool, index: int) -> None:
    """Log metrics to wandb if enabled."""
    if not wandb_enabled:
        return

    try:
        # Use Unix timestamp as step for consistent ordering
        timestamp_unix = time.time()

        wandb_metrics = {
            "eval/timestamp": timestamp_unix,
            f"eval/loss_{index}": metrics.get("loss"),
            f"eval/accuracy_{index}": metrics.get("accuracy"),
            f"eval/num_items_{index}": metrics.get("num_items"),
            f"eval/num_batches_{index}": metrics.get("num_batches"),
        }

        # Filter out None values
        wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}

        wandb.log(wandb_metrics)
        logger.info(f"Logged metrics to wandb: eval/loss_{index}={metrics.get('loss', 'N/A'):.4f}, eval/accuracy_{index}={metrics.get('accuracy', 'N/A'):.4f}")
    except Exception as e:
        logger.error(f"Failed to log to wandb: {e}")


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_pipeline_forward(models: List[torch.nn.Module], x, labels=None):
    """
    Run forward pass through all pipeline stages, mimicking BaselineModel.forward().
    Labels are only passed to the last stage.
    """
    num_stages = len(models)
    is_sequence = isinstance(x, (tuple, list))
    
    for idx, model in enumerate(models):
        is_last_stage = (idx == num_stages - 1)
        if is_sequence:
            if labels is not None and is_last_stage:
                x = model(*x, labels)
            else:
                x = model(*x)
            is_sequence = isinstance(x, (tuple, list))
        else:
            if labels is not None and is_last_stage:
                x = model(x, labels)
            else:
                x = model(x)
    return x


def run_evaluation(cfg: Config, models: List[torch.nn.Module]) -> Dict[str, Any]:
    """Run evaluation through full pipeline and return metrics."""
    timestamp = datetime.now().isoformat()
    dev = _resolve_device()
    task_type = cfg.data.task_type

    logger.info(f"[{timestamp}] Running evaluation")
    logger.info(f"  Config: {CONFIG_PATH}")
    logger.info(f"  Device: {dev}")
    logger.info(f"  Task type: {task_type}")
    logger.info(f"  Pipeline stages: {len(models)}")

    bs = int(cfg.diloco.batch_size_per_step)

    # Move all models to device
    for model in models:
        model.to(dev)
        model.eval()

    _, val_ds = get_train_val_datasets(cfg.data)
    cfn = collate_fn(cfg.data, cfg.model_pipeline.pipeline[0])
    loader = DataLoader(
        val_ds,
        batch_size=bs,
        num_workers=0,  # Avoid shared memory issues
        collate_fn=cfn,
    )

    total_loss = 0.0
    total_items = 0
    total_accuracy = 0.0

    # Determine mixed precision dtype for GPU
    amp_dtype = None
    if dev.type == "cuda":
        if cfg.data.precision == "fp16-mixed":
            amp_dtype = torch.float16
        elif cfg.data.precision == "bf16-mixed":
            amp_dtype = torch.bfloat16

    with torch.inference_mode():
        for i, (_uids, batch) in enumerate(loader):
            if i >= MAX_BATCHES:
                break

            inputs = batch["inputs"]
            labels = batch["labels"]

            if isinstance(inputs, tuple):
                inputs = tuple(x.to(dev) if hasattr(x, "to") else x for x in inputs)
            else:
                inputs = inputs.to(dev)
            labels = labels.to(dev) if hasattr(labels, "to") else labels

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None
                else torch.no_grad()
            )
            with autocast_ctx:
                # Run through all pipeline stages
                outputs = _run_pipeline_forward(models, inputs, labels)
                loss_t = task_type_loss(cfg, inputs, outputs, labels)

            # For node_pred, outputs contain [loss, accuracy]
            acc = outputs.squeeze(0)[1]
            total_loss += float(loss_t.item())
            total_accuracy += float(acc.item())
            total_items += 1

            if (i + 1) % 10 == 0:
                logger.info(f"  Batch {i + 1}/{MAX_BATCHES}, running loss: {total_loss / total_items:.4f}, accuracy: {total_accuracy / total_items:.4f}")

    # Move all models back to CPU and clean up GPU memory
    for model in models:
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if total_items == 0:
        return {"error": "No batches evaluated", "timestamp": timestamp}

    mean_loss = total_loss / total_items
    mean_accuracy = total_accuracy / total_items
    metrics: Dict[str, Any] = {
        "timestamp": timestamp,
        "split": SPLIT,
        "task_type": task_type,
        "loss": mean_loss,
        "accuracy": mean_accuracy,
        "num_items": total_items,
        "num_batches": min(MAX_BATCHES, total_items),
        "device": str(dev),
        "hostname": os.uname().nodename,
    }

    return metrics


def save_metrics(metrics: Dict[str, Any]) -> Path:
    """Save metrics to a timestamped JSON file."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = os.uname().nodename.replace(".", "_")
    output_file = EVAL_RESULTS_DIR / f"eval_{hostname}_{timestamp_str}.json"

    output_file.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    logger.info(f"Metrics saved to: {output_file}")

    # Also update a "latest" symlink/file for easy access
    latest_file = EVAL_RESULTS_DIR / f"eval_{hostname}_latest.json"
    latest_file.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    return output_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate GCN model with parameter mirroring")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Evaluation index for wandb logging (default: 0)",
    )
    parser.add_argument(
        "--wandb-run-id-path",
        type=str,
        default=None,
        help="Path to file containing wandb run ID (preferred over DHT lookup)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wandb_enabled = False
    dht = None
    param_mirror = None

    try:
        if not CONFIG_PATH.exists():
            error_metrics = {
                "timestamp": datetime.now().isoformat(),
                "error": f"Config not found: {CONFIG_PATH}",
                "hostname": os.uname().nodename,
            }
            save_metrics(error_metrics)
            logger.error(f"Config not found: {CONFIG_PATH}")
            return 1

        cfg = parse_yaml_file_as(Config, CONFIG_PATH)

        # Initialize DHT connection
        dht = _init_dht(cfg)
        if dht is None:
            return 1

        # Initialize parameter mirror
        param_mirror = EvalParamMirror(cfg, dht)
        models = param_mirror.get_all_models()
        if not models:
            logger.error("No models initialized from parameter mirror")
            return 1

        # Initialize wandb
        wandb_enabled = _init_wandb(cfg, dht, args.wandb_run_id_path)

        logger.info("Starting evaluation")

        # Refresh parameters before evaluation
        try:
            refresh_success = param_mirror.refresh_parameters()
            if not refresh_success:
                logger.warning("Parameter refresh failed, using existing parameters for this evaluation")
        except RuntimeError as e:
            logger.error(f"Fatal error: {e}")
            error_metrics = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "hostname": os.uname().nodename,
            }
            save_metrics(error_metrics)
            return 1

        metrics = run_evaluation(cfg, models)
        save_metrics(metrics)
        logger.info(json.dumps(metrics, indent=2, sort_keys=True))

        if "error" in metrics:
            logger.error(f"Evaluation failed: {metrics['error']}")
            return 1
        else:
            _log_to_wandb(metrics, wandb_enabled, args.index)
            logger.info(f"Evaluation completed. Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        return 0

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 0

    except Exception as e:
        error_metrics = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "hostname": os.uname().nodename,
        }
        save_metrics(error_metrics)
        logger.error(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if wandb_enabled:
            try:
                wandb.finish()
            except Exception:
                pass
        if dht is not None:
            try:
                dht.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    exit(main())
