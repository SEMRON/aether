#!/usr/bin/env python3
"""
Periodic evaluation script for distributed training workers.
Mirrors parameters from workers via DiLoCo optimizer and runs evaluation on
the validation split. Logs metrics to wandb and JSON.
Joins the same wandb run as monitor/workers via DHT-shared run ID.
"""

# NOTE: We intentionally do NOT set multiprocessing.set_start_method('spawn') here.
# While 'spawn' fixes CUDA cleanup issues with forked processes, it breaks hivemind's
# DecentralizedAverager because internal objects (SimpleQueue, etc.) cannot be pickled.

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
# Use file_system sharing strategy to reduce shared memory issues
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
from hivemind.dht import DHT
from hivemind.moe.server.layers import name_to_block
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from pydantic_yaml import parse_yaml_file_as
from torch.utils.data import DataLoader

from distqat.config import Config
from distqat.data import collate_fn
from distqat.models import kwargs_from_config
from distqat.optimizers import get_diloco_optimizer_cls_kwargs
from distqat.utils.compression import get_compression_kwargs
from distqat.utils.logging import get_wandb_run_id_with_retries
from distqat.utils.loss import task_type_loss
from distqat.data import get_train_val_datasets


logger = get_logger(__name__)
use_hivemind_log_handler("in_root_logger")

# Configuration
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "biggan_cifar.yaml"
INITIAL_PEERS_PATH = Path(__file__).resolve().parents[3] / "logs" / "biggan_cifar" / "initial_peers.txt"
EVAL_RESULTS_DIR = Path(__file__).resolve().parents[3] / "logs" / "biggan_cifar" / "eval_results"
CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints" / "biggan_cifar_eval"
MAX_BATCHES = 20
EVAL_INTERVAL = 60
SAVE_CHECKPOINTS = True
KEEP_HISTORY = False


class EvalParamMirror:
    """
    Mirrors parameters from workers via DiLoCo optimizer.
    Uses the same optimizer as training servers to ensure protocol compatibility.
    """

    # Maximum consecutive refresh failures before requiring restart
    MAX_CONSECUTIVE_FAILURES = 5
    # Timeout for loading state (5 minutes for large models like BigGAN)
    LOAD_STATE_TIMEOUT = 300.0
    # Number of retries for initial state load
    INITIAL_LOAD_RETRIES = 5
    # Delay between retries (seconds)
    RETRY_DELAY = 10.0

    def __init__(self, cfg: Config, dht: DHT):
        cfg.biggan['enable_eval'] = True
        self.cfg = cfg
        self.dht = dht
        self._mirrors: List[Tuple[torch.nn.Module, object, str, int]] = []  # model, optimizer, run_id, stage_index
        self._compression = get_compression_kwargs(cfg.network.hivemind_compression)
        self._consecutive_failures = 0
        self._state_loaded = False  # Track whether we successfully loaded state
        self._last_outer_steps: Dict[int, int] = {}

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

            aliases = {"config": cfg.biggan}
            model_kwargs = kwargs_from_config(block_ctor.__init__, pipeline_step_cfg, cfg.data, aliases)
            model = block_ctor(**model_kwargs)
            model.to("cpu")
            model.eval()

            run_id = f"{cfg.experiment_prefix}_{stage_index}"
            optimizer = self._create_optimizer(model, run_id)
            if optimizer is None:
                continue

            # Initial parameter sync from peers with retries
            loaded = self._load_state_with_retries(model, optimizer, stage_index)
            if loaded:
                self._clone_model_parameters(model)
                self._state_loaded = True
            else:
                logger.error(f"Failed to load initial state for stage {stage_index} after {self.INITIAL_LOAD_RETRIES} attempts")

            self._mirrors.append((model, optimizer, run_id, stage_index))
            self._last_outer_steps[stage_index] = int(getattr(optimizer, "outer_step", 0) or 0)

        gc.collect()
        logger.info(f"EvalParamMirror initialized with {len(self._mirrors)} stage(s), state_loaded={self._state_loaded}")

    def _load_state_with_retries(self, model: torch.nn.Module, optimizer: object, stage_index: int) -> bool:
        """
        Load state from peers with retries.
        Returns True if state was successfully loaded, False otherwise.
        """
        # Guardrail: ensure there is at least one peer reporting progress for this stage.
        # Otherwise, load_state_from_peers can be a no-op without raising.
        self._wait_for_training_progress(stage_index, timeout_s=min(30.0, self.LOAD_STATE_TIMEOUT))

        before_sig = self._param_signature(model)
        for attempt in range(1, self.INITIAL_LOAD_RETRIES + 1):
            try:
                logger.info(f"Loading initial state for stage {stage_index} (attempt {attempt}/{self.INITIAL_LOAD_RETRIES})...")
                self._load_state_with_timeout(optimizer)
                after_sig = self._param_signature(model)
                # If params didn't change after a "successful" load, it may be a no-op.
                # However, at the very beginning of training, peers can share identical initialization.
                # Only enforce this if we see evidence of real training progress for this stage.
                max_outer, max_inner = self._max_reported_progress(stage_index)
                optimizer_outer = int(getattr(optimizer, "outer_step", 0) or 0)
                progressed = (max_outer > 0) or (max_inner > 0) or (optimizer_outer > 0)
                if progressed and before_sig is not None and after_sig is not None and after_sig == before_sig:
                    raise RuntimeError(
                        "load_state_from_peers appeared to be a no-op (parameters unchanged despite training progress)"
                    )
                logger.info(f"Successfully loaded initial state for stage {stage_index}")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{self.INITIAL_LOAD_RETRIES} failed for stage {stage_index}: {e}")
                if attempt < self.INITIAL_LOAD_RETRIES:
                    logger.info(f"Retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)
        return False

    def has_loaded_state(self) -> bool:
        """Returns True if state was successfully loaded from peers."""
        return self._state_loaded

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
                expert=model,
                **optim_kwargs,
            )
        except Exception as e:
            logger.warning(f"EvalParamMirror: failed to create optimizer for {run_id}: {e}")
            return None

    def _load_state_with_timeout(self, optimizer: object) -> None:
        """
        Load state from peers with controlled timeout.
        Uses direct averager call to avoid the optimizer's infinite retry loop.
        """
        if hasattr(optimizer, 'averager'):
            optimizer.averager.load_state_from_peers(timeout=self.LOAD_STATE_TIMEOUT)
        else:
            optimizer.load_state_from_peers()

    def _wait_for_training_progress(self, stage_index: int, timeout_s: float) -> bool:
        """Wait briefly until at least one peer reports progress for this stage in the DHT."""
        key = f"{self.cfg.experiment_prefix}_{stage_index}_progress"
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            try:
                resp = self.dht.get(key, latest=True)
                if resp is not None and hasattr(resp, "value"):
                    val = resp.value
                else:
                    val = resp
                if isinstance(val, dict):
                    # DHT returns {subkey -> ValueWithExpiration}; accept any non-null value.
                    for entry in val.values():
                        if hasattr(entry, "value") and entry.value is not None:
                            return True
            except Exception:
                pass
            time.sleep(1.0)
        return False

    def _max_reported_progress(self, stage_index: int) -> Tuple[int, int]:
        """
        Best-effort read of (max_outer_step, max_inner_step) for this stage from DHT progress.
        Returns (0, 0) if no progress is found or parsing fails.
        """
        key = f"{self.cfg.experiment_prefix}_{stage_index}_progress"
        try:
            resp = self.dht.get(key, latest=True)
            if resp is None:
                return (0, 0)
            val = resp.value if hasattr(resp, "value") else resp
            if not isinstance(val, dict):
                return (0, 0)
            max_outer, max_inner = 0, 0
            for entry in val.values():
                if not hasattr(entry, "value") or entry.value is None:
                    continue
                payload = entry.value
                if isinstance(payload, dict):
                    max_outer = max(max_outer, int(payload.get("outer_step", 0) or 0))
                    max_inner = max(max_inner, int(payload.get("inner_step", 0) or 0))
            return (max_outer, max_inner)
        except Exception:
            return (0, 0)

    def _param_signature(self, model: torch.nn.Module) -> Optional[Tuple[float, int]]:
        """
        Compute a cheap signature of model parameters to detect no-op refreshes.
        Returns (sum_abs, numel) over a small subset of parameters.
        """
        try:
            total = 0.0
            numel = 0
            with torch.no_grad():
                for i, p in enumerate(model.parameters()):
                    if i >= 8:
                        break
                    t = p.detach().float().cpu()
                    total += float(t.abs().sum().item())
                    numel += int(t.numel())
            return (total, numel)
        except Exception:
            return None

    def _clone_model_parameters(self, model: torch.nn.Module) -> None:
        """Clone all model parameters in-place to avoid shared memory backing."""
        with torch.no_grad():
            for param in model.parameters():
                param.data = param.data.clone().detach()
            for buffer in model.buffers():
                buffer.data = buffer.data.clone().detach()
        gc.collect()

    def refresh_parameters(self) -> bool:
        """
        Sync parameters from workers. Returns True if at least one stage was synced.
        Raises RuntimeError if MAX_CONSECUTIVE_FAILURES is exceeded.
        """
        success = False
        for model, optimizer, run_id, stage_index in self._mirrors:
            try:
                logger.debug(f"Loading state from peers for {run_id}...")
                self._load_state_with_timeout(optimizer)
                self._clone_model_parameters(model)
                success = True
                self._last_outer_steps[stage_index] = int(getattr(optimizer, "outer_step", 0) or 0)
                logger.debug(f"Refreshed parameters for {run_id}")
            except Exception as e:
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

        gc.collect()
        return success

    def get_outer_steps(self) -> Dict[int, int]:
        """Return last observed outer_step per stage (as mirrored by this evaluator)."""
        return dict(self._last_outer_steps)

    def get_model(self) -> Optional[torch.nn.Module]:
        """Returns the first stage model for evaluation."""
        if not self._mirrors:
            return None
        return self._mirrors[0][0]

    def get_all_models(self) -> List[torch.nn.Module]:
        """Returns all stage models."""
        return [model for model, _, _, _ in self._mirrors]

    def shutdown(self) -> None:
        """Shutdown all optimizers."""
        for model, optimizer, run_id, stage_index in self._mirrors:
            try:
                if hasattr(optimizer, 'shutdown'):
                    optimizer.shutdown()
            except Exception:
                pass


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
    """Initialize wandb for logging evaluation metrics. Returns True if successful."""
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
        # Define eval metrics with their own step counter
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        logger.info(f"Wandb initialized with shared run: {cfg.wandb_project}/{wandb_run_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        return False


def _log_to_wandb(metrics: Dict[str, Any], wandb_enabled: bool, eval_step: int) -> None:
    """Log metrics to wandb if enabled."""
    if not wandb_enabled:
        return

    try:
        wandb_metrics = {
            "eval/step": eval_step,
            "eval/loss": metrics.get("loss"),
            "eval/num_items": metrics.get("num_items"),
        }
        if "perplexity" in metrics:
            wandb_metrics["eval/perplexity"] = metrics["perplexity"]
        if "accuracy_top1" in metrics:
            wandb_metrics["eval/accuracy"] = metrics["accuracy_top1"]
        
        # Image generation metrics (IS/FID)
        if "IS_mean" in metrics:
            wandb_metrics["eval/IS_mean"] = metrics["IS_mean"]
        if "IS_std" in metrics:
            wandb_metrics["eval/IS_std"] = metrics["IS_std"]
        if "FID" in metrics:
            wandb_metrics["eval/FID"] = metrics["FID"]
        if "best_IS" in metrics:
            wandb_metrics["eval/best_IS"] = metrics["best_IS"]
        if "best_FID" in metrics:
            wandb_metrics["eval/best_FID"] = metrics["best_FID"]

        wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
        wandb.log(wandb_metrics, commit=True)
        
        # Log appropriate message based on task type
        if "IS_mean" in metrics:
            logger.info(f"Logged metrics to wandb: IS={metrics.get('IS_mean', 'N/A'):.3f}, FID={metrics.get('FID', 'N/A'):.4f} at step {eval_step}")
        else:
            logger.info(f"Logged metrics to wandb: eval/loss={metrics.get('loss', 'N/A'):.4f} at step {eval_step}")
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


def run_evaluation(cfg: Config, models: List[torch.nn.Module], eval_step: int = 0) -> Dict[str, Any]:
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

    # For image generation (BigGAN), use the model's built-in IS/FID evaluation
    if task_type == "image_gen":
        metrics = _run_image_gen_evaluation(cfg, models, dev, timestamp, eval_step)
        # Move models back to CPU
        for model in models:
            model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    ds = get_train_val_datasets(cfg.data)[1]
    cfn = collate_fn(cfg.data, cfg.model_pipeline.pipeline[0])
    loader = DataLoader(
        ds,
        batch_size=bs,
        num_workers=0,  # Avoid shared memory issues
        collate_fn=cfn,
    )

    total_loss = 0.0
    total_items = 0
    total_correct = 0

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

            bsz = int(labels.shape[0]) if hasattr(labels, "shape") else bs
            total_loss += float(loss_t.item()) * bsz
            total_items += bsz

            # Track accuracy for CV tasks
            if task_type == "cv":
                preds = outputs.argmax(dim=-1)
                total_correct += int((preds == labels).sum().item())

            if (i + 1) % 10 == 0:
                logger.info(f"  Batch {i + 1}/{MAX_BATCHES}, running loss: {total_loss / total_items:.4f}")

    # Move all models back to CPU and clean up GPU memory
    for model in models:
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if total_items == 0:
        return {"error": "No batches evaluated", "timestamp": timestamp}

    mean_loss = total_loss / total_items
    metrics: Dict[str, Any] = {
        "timestamp": timestamp,
        "split": cfg.data.dataset_split_validation,
        "task_type": task_type,
        "loss": mean_loss,
        "num_items": total_items,
        "num_batches": min(MAX_BATCHES, math.ceil(total_items / bs)),
        "device": str(dev),
        "hostname": os.uname().nodename,
    }

    # Add task-specific metrics
    if task_type == "cv":
        metrics["accuracy_top1"] = total_correct / total_items
    if task_type == "llm":
        metrics["perplexity"] = float(math.exp(mean_loss)) if mean_loss < 50 else float("inf")

    return metrics


def _run_image_gen_evaluation(
    cfg: Config, 
    models: List[torch.nn.Module], 
    dev: torch.device, 
    timestamp: str,
    eval_step: int
) -> Dict[str, Any]:
    """
    Run evaluation for image generation models (BigGAN) using Inception Score and FID.
    Uses the model's built-in evaluate() method.
    """
    metrics: Dict[str, Any] = {
        "timestamp": timestamp,
        "task_type": "image_gen",
        "device": str(dev),
        "hostname": os.uname().nodename,
    }

    # BigGAN is typically a single-stage model
    if len(models) != 1:
        logger.warning(f"Expected 1 model for image_gen evaluation, got {len(models)}")
    
    model = models[0]
    
    # Check if model has the evaluate method (BigGANAdapter)
    if not hasattr(model, 'evaluate'):
        logger.error("Model does not have evaluate() method for IS/FID calculation")
        metrics["error"] = "Model missing evaluate() method"
        return metrics
    
    # Check if evaluation is enabled in the model
    if hasattr(model, 'enable_eval') and not model.enable_eval:
        logger.warning("Model evaluation is disabled (enable_eval=False)")
        metrics["error"] = "Model evaluation disabled"
        return metrics
    
    logger.info(f"Running IS/FID evaluation at step {eval_step}...")
    
    try:
        eval_results = model.evaluate(eval_step)
        
        if eval_results is None:
            logger.warning("Model evaluate() returned None - check inception network and moments file")
            metrics["error"] = "Evaluation returned None"
            return metrics
        
        # Add IS/FID metrics (convert numpy types to Python floats for JSON serialization)
        metrics["IS_mean"] = float(eval_results.get("IS_mean"))
        metrics["IS_std"] = float(eval_results.get("IS_std"))
        metrics["FID"] = float(eval_results.get("FID"))
        metrics["best_IS"] = float(eval_results.get("best_IS"))
        metrics["best_FID"] = float(eval_results.get("best_FID"))
        
        logger.info(
            f"Evaluation complete: IS={metrics['IS_mean']:.3f}±{metrics['IS_std']:.3f}, "
            f"FID={metrics['FID']:.4f}"
        )
        
    except Exception as e:
        logger.error(f"IS/FID evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        metrics["error"] = str(e)
    
    return metrics


def save_metrics(metrics: Dict[str, Any]) -> Path:
    """Save metrics to a timestamped JSON file."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = os.uname().nodename.replace(".", "_")
    output_file = EVAL_RESULTS_DIR / f"eval_{hostname}_{timestamp_str}.json"

    output_file.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    logger.info(f"Metrics saved to: {output_file}")

    # Also update a "latest" file for easy access
    latest_file = EVAL_RESULTS_DIR / f"eval_{hostname}_latest.json"
    latest_file.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    return output_file


def save_checkpoint(models: List[torch.nn.Module], checkpoint_dir: Path, keep_history: bool = True) -> Optional[Path]:
    """
    Save a checkpoint combining all pipeline stages in BaselineModel-compatible format.
    
    The checkpoint is saved with keys like 'model_pipeline.0.xxx', 'model_pipeline.1.xxx', etc.
    so it can be loaded directly by BaselineModel.
    
    Args:
        models: List of pipeline stage models
        checkpoint_dir: Directory to save checkpoints
        keep_history: If True, save timestamped checkpoints; if False, only keep latest
        
    Returns:
        Path to the saved checkpoint, or None if saving failed
    """
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all stage state_dicts with model_pipeline.X. prefix
        combined_state_dict = {}
        for stage_idx, model in enumerate(models):
            stage_state = model.state_dict()
            for key, value in stage_state.items():
                combined_key = f"model_pipeline.{stage_idx}.{key}"
                combined_state_dict[combined_key] = value
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if keep_history:
            checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}.pt"
            tmp_path = checkpoint_dir / f".checkpoint_{timestamp}.pt.tmp"
        else:
            checkpoint_path = checkpoint_dir / "checkpoint_last.pt"
            tmp_path = checkpoint_dir / ".checkpoint_last.pt.tmp"
        
        # Save to temp file first, then atomic rename
        torch.save(combined_state_dict, tmp_path)
        os.replace(tmp_path, checkpoint_path)
        
        # Update/create checkpoint_last.pt symlink
        if keep_history:
            last_path = checkpoint_dir / "checkpoint_last.pt"
            tmp_last = checkpoint_dir / ".checkpoint_last.pt.tmp"
            try:
                tmp_last.unlink(missing_ok=True)
            except Exception:
                pass
            os.symlink(checkpoint_path.name, tmp_last)
            os.replace(tmp_last, last_path)
            
            # Also clean up old checkpoints (keep last 5)
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
            checkpoints = [c for c in checkpoints if c.name != "checkpoint_last.pt"]
            if len(checkpoints) > 5:
                for old_ckpt in checkpoints[:-5]:
                    try:
                        old_ckpt.unlink()
                    except Exception:
                        pass
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model with parameter mirroring from distributed training")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL,
        help=f"Interval between evaluations in seconds (default: {EVAL_INTERVAL})",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Run a single evaluation and exit (default: continuous loop)",
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
        # Ensure BigGANAdapter picks a device that exists on this evaluator host.
        # (This does not affect state mirroring, since device is not part of state_dict.)
        cfg.biggan["device"] = str(_resolve_device())

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
        
        # Verify state was actually loaded from peers
        if not param_mirror.has_loaded_state():
            logger.error("Failed to load model state from peers - cannot evaluate with random weights")
            error_metrics = {
                "timestamp": datetime.now().isoformat(),
                "error": "Failed to load model state from peers after multiple retries",
                "hostname": os.uname().nodename,
            }
            save_metrics(error_metrics)
            return 1
        
        logger.info(f"Loaded {len(models)} pipeline stages for evaluation")

        # Initialize wandb
        wandb_enabled = _init_wandb(cfg, dht, args.wandb_run_id_path)

        eval_count = 0

        while True:
            logger.info(f"Starting evaluation #{eval_count + 1}")

            # Refresh parameters before each evaluation
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

            outer_steps = {}
            try:
                outer_steps = param_mirror.get_outer_steps()
            except Exception:
                outer_steps = {}
            eval_step = int(outer_steps.get(0, eval_count) or eval_count)

            metrics = run_evaluation(cfg, models, eval_step=eval_step)
            metrics["mirrored_outer_step_stage0"] = outer_steps.get(0, None)
            save_metrics(metrics)
            logger.info(json.dumps(metrics, indent=2, sort_keys=True))

            if "error" in metrics:
                logger.error(f"Evaluation failed: {metrics['error']}")
            else:
                _log_to_wandb(metrics, wandb_enabled, eval_step)
                if "IS_mean" in metrics:
                    logger.info(f"Evaluation #{eval_count + 1} completed. IS: {metrics['IS_mean']:.3f}, FID: {metrics['FID']:.4f}")
                else:
                    logger.info(f"Evaluation #{eval_count + 1} completed. Loss: {metrics['loss']:.4f}")
                
                # Save checkpoint after successful evaluation
                if SAVE_CHECKPOINTS:
                    save_checkpoint(models, CHECKPOINT_DIR, keep_history=KEEP_HISTORY)

            eval_count += 1

            if args.one_shot:
                break

            time.sleep(args.eval_interval)

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
        if param_mirror is not None:
            try:
                param_mirror.shutdown()
            except Exception:
                pass
        if dht is not None:
            try:
                dht.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    exit(main())
