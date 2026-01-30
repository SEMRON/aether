import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import click
import time
import signal
import json
import math
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

from hivemind.dht import DHT
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from distqat.config import Config, parse_args
from distqat.distributed.param_mirror import ParamMirror
from distqat.distributed.model import BaselineModel
from distqat.utils import logging
from distqat.data import get_train_val_datasets, get_dataloader
from torch.utils.data import DataLoader

logger = get_logger(__name__)
logger.setLevel('DEBUG')


class Evaluator:
    """
    Distributed evaluator that mirrors parameters from training peers
    and runs periodic evaluation on validation data.
    
    Similar to SwarmTrainer but operates in eval-only mode:
    - Uses ParamMirror to sync parameters from distributed training (default)
    - Or loads from baseline checkpoint on disk (baseline=True)
    - Runs forward passes on validation data (no gradients)
    - Computes task-specific metrics (accuracy, perplexity, etc.)
    - Logs results to wandb and files at regular intervals
    """
    
    def __init__(self, cfg: Config, eval_interval: int = 60, baseline: bool = False):
        self.cfg = cfg
        self.disable_quant = cfg.disable_quant
        self.eval_interval = eval_interval
        self.device = cfg.device
        self.stop_requested = False
        self.stop_event = threading.Event()  # For interruptible sleep
        self.baseline = baseline
        
        # For image generation (BigGAN), ensure evaluation is enabled
        # This is required for IS/FID computation via model.evaluate()
        if cfg.data.task_type == "image_gen":
            if hasattr(cfg, 'biggan') and isinstance(cfg.biggan, dict):
                cfg.biggan['enable_eval'] = True
                logger.info("Enabled BigGAN evaluation (enable_eval=True) for IS/FID computation")
            # Also update pipeline extra configs - Pydantic may create separate dict copies
            # from YAML anchors, so cfg.biggan and pipeline[].extra may be different objects
            if hasattr(cfg, 'model_pipeline') and hasattr(cfg.model_pipeline, 'pipeline'):
                for step_cfg in cfg.model_pipeline.pipeline:
                    if hasattr(step_cfg, 'extra') and isinstance(step_cfg.extra, dict):
                        if 'enable_eval' in step_cfg.extra:
                            step_cfg.extra['enable_eval'] = True
                            logger.debug(f"Also enabled enable_eval in pipeline step: {step_cfg.model_name}")
        
        # Initialize DHT
        self.dht = DHT(
            start=True,
            initial_peers=cfg.network.initial_peers,
            host_maddrs=cfg.network.host_maddrs,
            announce_maddrs=cfg.network.announce_maddrs
        )

        # Setup file logging
        log_suffix = "baseline_evaluator" if baseline else "evaluator"
        log_file = cfg.log_dir / f"{log_suffix}.log"
        logging.setup_file_logging(log_file, wandb_enabled=False)

        # Setup DHT-based metrics publishing (monitor will log to wandb)
        self._setup_metrics_publishing(cfg)

        run_id = f"{cfg.experiment_prefix}_{log_suffix}"
        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        logger.info(f"Running DHT node on {visible_maddrs_str}")
        logger.debug(f"Initial peers: {cfg.network.initial_peers}")
        logger.info(f"====> RUN_ID: {run_id}")
        logger.info(f"Mode: {'baseline' if baseline else 'distributed'}")

        if baseline:
            # Baseline mode: load from disk checkpoint
            self.param_mirror = None
            self._checkpoint_path = self._get_baseline_checkpoint_path()
            self._checkpoint_mtime: Optional[float] = None
            logger.info(f"Baseline evaluator will load from: {self._checkpoint_path}")
            
            # Create and load baseline model
            self._create_baseline_model()
        else:
            # Distributed mode: use ParamMirror to sync parameters from training peers
            # Initialize ParamMirror to sync parameters from training peers
            self.param_mirror = ParamMirror(cfg, self.dht, refresh_every=min(eval_interval, 30))
            self.param_mirror.start()
            
            # Wait a bit for initial parameter sync
            logger.info("Waiting for initial parameter sync from peers...")
            time.sleep(5)

            
            # Create local evaluation model (copy of ParamMirror's model structure)
            # This avoids race conditions and device conflicts with the ParamMirror thread
            self._create_eval_model()
        
        # Track evaluation step for logging
        self.eval_step = 0
        
        # Create validation dataloader once (like trainer pattern)
        # Only recreate when exhausted or on errors
        self._init_val_dataloader()
        
        logger.info(f"Evaluator initialized with eval_interval={eval_interval}s")
        logger.info(f"Task type: {cfg.data.task_type}")
    
    def _get_baseline_checkpoint_path(self) -> Path:
        """Get the path to the baseline checkpoint."""
        if self.cfg.checkpoint_dir is None:
            raise ValueError("Config.checkpoint_dir must be set for baseline evaluation")
        return Path(self.cfg.checkpoint_dir) / "baseline" / "checkpoint_last.pt"
    
    def _create_baseline_model(self):
        """Create the baseline model and load weights from checkpoint."""
        from distqat.attach import attach_quantizers
        
        model = BaselineModel(self.cfg)
        if not self.disable_quant:
            model, _ = attach_quantizers(model, self.cfg.quant)
        model.to(self.device)
        model.eval()
        
        # Store as list to match the distributed evaluator interface
        self._eval_models = [model]
        self._sync_lock = threading.Lock()
        
        # Load initial weights if checkpoint exists
        self._load_baseline_checkpoint()
        
        logger.info(f"Created baseline evaluation model on {self.device}")
    
    def _load_baseline_checkpoint(self) -> bool:
        """
        Load weights from the baseline checkpoint file.
        
        Returns True if checkpoint was reloaded, False if no changes detected.
        """
        if not self._checkpoint_path.exists():
            logger.warning(f"Baseline checkpoint not found at {self._checkpoint_path}, using random weights")
            return False
        
        # Check if checkpoint has been modified since last load
        try:
            current_mtime = self._checkpoint_path.stat().st_mtime
        except FileNotFoundError:
            return False
        
        if self._checkpoint_mtime is not None and current_mtime <= self._checkpoint_mtime:
            logger.debug("Baseline checkpoint unchanged, skipping reload")
            return False
        
        try:
            obj = torch.load(self._checkpoint_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
                state_dict = obj["model"]
            elif isinstance(obj, dict):
                state_dict = obj
            else:
                logger.error(f"Unexpected checkpoint payload type: {type(obj)}")
                return False
            
            # Load into the model
            with self._sync_lock:
                # The BaselineModel wraps stages in model_pipeline, so load directly
                missing, unexpected = self._eval_models[0].load_state_dict(state_dict, strict=False)
                if missing:
                    logger.debug(f"Missing keys: {missing[:5]}...")
                if unexpected:
                    logger.debug(f"Unexpected keys: {unexpected[:5]}...")
            
            self._checkpoint_mtime = current_mtime
            logger.info(f"Loaded baseline checkpoint from {self._checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load baseline checkpoint: {e}")
            return False

    def _setup_metrics_publishing(self, cfg: Config):
        """Setup metrics publishing to DHT for the monitor to collect.
        
        Instead of logging directly to wandb (which causes step conflicts when
        multiple processes share a run), we publish metrics to DHT and let
        the monitor log everything to wandb centrally - same pattern as trainers.
        """
        from hivemind.utils import get_dht_time
        
        # Use different DHT keys for baseline vs distributed evaluation
        key_suffix = "_baseline_eval_metrics" if self.baseline else "_eval_metrics"
        self._dht_key = cfg.experiment_prefix + key_suffix
        self._dht_subkey = self.dht.peer_id.to_bytes()
        self._metrics_expiration = 300.0  # 5 minutes
        self._get_dht_time = get_dht_time
        logger.info(f"Evaluator will publish metrics to DHT key: {self._dht_key}")

    def _init_val_dataloader(self):
        """Initialize the validation dataloader (created once, reused across evaluations)."""
        task_type = self.cfg.data.task_type
        
        # Skip for RL - requires environment interaction
        if task_type == "rl":
            self.val_dataloader = None
            self._val_dataloader_iter = None
            return
        
        _, val_ds = get_train_val_datasets(self.cfg.data)
        self.val_dataloader = get_dataloader(self.cfg, val_ds)
        self._val_dataloader_iter = iter(self.val_dataloader)
        logger.info("Initialized validation dataloader")

    def _fetch_batch_with_retry(self):
        """
        Fetch a batch from the validation dataloader, handling exhaustion and errors.
        Similar to trainer pattern - only recreates dataloader when necessary.
        
        Returns:
            (uid, batch) tuple, or None if dataloader is unavailable
        """
        if self.val_dataloader is None:
            return None
        
        try:
            t0 = time.time()
            uid, batch = next(self._val_dataloader_iter)
            dt = time.time() - t0
            logger.debug(f"[EVALUATOR:DataLoader] Fetch time: {dt:.4f}s")
            return uid, batch
        except StopIteration:
            logger.info("Validation dataset exhausted, recreating dataloader for next epoch")
            _, val_ds = get_train_val_datasets(self.cfg.data)
            self.val_dataloader = get_dataloader(self.cfg, val_ds)
            self._val_dataloader_iter = iter(self.val_dataloader)
            t0 = time.time()
            uid, batch = next(self._val_dataloader_iter)
            dt = time.time() - t0
            logger.debug(f"[EVALUATOR:DataLoader] Fetch time (reload): {dt:.4f}s")
            return uid, batch
        except (FileNotFoundError, RuntimeError) as e:
            error_str = str(e).lower()
            # Handle FileNotFoundError from missing parquet files in Hugging Face datasets
            if isinstance(e, FileNotFoundError) or "filenotfound" in error_str:
                logger.warning(f"FileNotFoundError in DataLoader (missing dataset file), recreating dataloader: {str(e)[:200]}")
                time.sleep(1)  # Brief delay before retry
                _, val_ds = get_train_val_datasets(self.cfg.data)
                self.val_dataloader = get_dataloader(self.cfg, val_ds)
                self._val_dataloader_iter = iter(self.val_dataloader)
                t0 = time.time()
                uid, batch = next(self._val_dataloader_iter)
                dt = time.time() - t0
                logger.debug(f"[EVALUATOR:DataLoader] Fetch time (after FileNotFoundError): {dt:.4f}s")
                return uid, batch
            # Handle DataLoader worker crashes
            elif isinstance(e, RuntimeError) and "DataLoader worker" in str(e) and ("exited unexpectedly" in str(e) or "is killed" in str(e)):
                logger.warning(f"DataLoader worker crashed, recreating dataloader: {e}")
                _, val_ds = get_train_val_datasets(self.cfg.data)
                self.val_dataloader = get_dataloader(self.cfg, val_ds)
                self._val_dataloader_iter = iter(self.val_dataloader)
                t0 = time.time()
                uid, batch = next(self._val_dataloader_iter)
                dt = time.time() - t0
                logger.debug(f"[EVALUATOR:DataLoader] Fetch time (after worker crash): {dt:.4f}s")
                return uid, batch
            else:
                raise

    def _create_eval_model(self):
        """Create a local copy of the model for evaluation.
        
        This avoids race conditions with the ParamMirror thread which updates
        the source models on CPU while we need them on GPU for evaluation.
        
        We create fresh model instances instead of using deepcopy() because some
        models (e.g., BigGAN) contain custom tensor subclasses or complex object
        graphs that don't survive deepcopy properly.
        """
        from hivemind.moe.server.layers import name_to_block
        from distqat.models import kwargs_from_config
        
        source_models = self.param_mirror.get_all_models()
        if not source_models:
            logger.warning("No models available from ParamMirror to create eval model")
            self._eval_models = []
            return
        
        # Lock for thread-safe sync operations
        self._sync_lock = threading.Lock()
        
        self._eval_models = []
        
        # Get model configuration from pipeline config
        pipeline_configs = list(self.cfg.model_pipeline.pipeline)
        
        for stage_index, source_model in enumerate(source_models):
            eval_model = None
            
            # Try to create a fresh model instance using the config
            if stage_index < len(pipeline_configs):
                try:
                    pipeline_step_cfg = pipeline_configs[stage_index]
                    model_name, stage_name = pipeline_step_cfg.model_name.split(".")
                    expert_cls = f"{model_name}.{stage_name}"
                    block_ctor = name_to_block.get(expert_cls)
                    
                    if block_ctor is not None:
                        aliases = {"config": pipeline_step_cfg.extra} if len(pipeline_step_cfg.extra.keys()) > 0 else None
                        model_kwargs = kwargs_from_config(block_ctor.__init__, pipeline_step_cfg, self.cfg.data, aliases=aliases)
                        eval_model = block_ctor(**model_kwargs)
                        
                        # Copy weights from source model
                        state_dict = source_model.state_dict()
                        eval_model.load_state_dict(state_dict, strict=False)
                        logger.debug(f"Created fresh eval model instance for stage {stage_index}")
                except Exception as e:
                    logger.warning(f"Failed to create fresh model for stage {stage_index}: {e}, falling back to deepcopy")
                    eval_model = None
            
            # Fallback to deepcopy if fresh instantiation failed
            if eval_model is None:
                from copy import deepcopy
                try:
                    eval_model = deepcopy(source_model)
                    logger.debug(f"Used deepcopy for eval model stage {stage_index}")
                except Exception as e:
                    logger.error(f"Failed to create eval model for stage {stage_index}: {e}")
                    continue
            
            eval_model.to(self.device)
            eval_model.eval()
            self._eval_models.append(eval_model)
        
        logger.info(f"Created {len(self._eval_models)} evaluation model(s) on {self.device}")

    def _sync_eval_model(self):
        """Sync evaluation model weights from ParamMirror's models.
        
        This copies the latest weights from ParamMirror (which runs on CPU)
        to our local evaluation model (which runs on GPU).
        """
        source_models = self.param_mirror.get_all_models()
        
        if len(source_models) != len(self._eval_models):
            logger.warning("Model count mismatch, recreating eval models")
            self._create_eval_model()
            return
        
        with self._sync_lock:
            for src_model, eval_model in zip(source_models, self._eval_models):
                # Get state dict from source (CPU) - this creates a copy
                state_dict = src_model.state_dict()
                
                # Debug: Check if BatchNorm buffers are in state_dict
                # Standard PyTorch BN uses running_mean/running_var
                # BigGAN's custom BN uses stored_mean/stored_var
                bn_keys = [k for k in state_dict.keys() if 'running_mean' in k or 'running_var' in k]
                biggan_bn_keys = [k for k in state_dict.keys() if 'stored_mean' in k or 'stored_var' in k]
                if self.eval_step < 3:
                    logger.debug(f"State dict has {len(bn_keys)} standard BN buffers (running_*), "
                                f"{len(biggan_bn_keys)} BigGAN BN buffers (stored_*)")
                    # Check standard BN buffers
                    if bn_keys:
                        first_rv_key = [k for k in bn_keys if 'running_var' in k][0] if any('running_var' in k for k in bn_keys) else None
                        if first_rv_key:
                            rv = state_dict[first_rv_key]
                            logger.debug(f"Source BN '{first_rv_key}': min={rv.min():.6f}, max={rv.max():.6f}, "
                                        f"near_zero={(rv < 1e-10).sum().item()}, negative={(rv < 0).sum().item()}")
                    # Check BigGAN BN buffers
                    if biggan_bn_keys:
                        first_sv_key = [k for k in biggan_bn_keys if 'stored_var' in k][0] if any('stored_var' in k for k in biggan_bn_keys) else None
                        if first_sv_key:
                            sv = state_dict[first_sv_key]
                            logger.debug(f"Source BigGAN BN '{first_sv_key}': min={sv.min():.6f}, max={sv.max():.6f}, "
                                        f"near_zero={(sv < 1e-10).sum().item()}, negative={(sv < 0).sum().item()}")
                
                # Move tensors to the eval model's device if needed
                device = next(eval_model.parameters()).device
                state_dict = {k: v.to(device) for k, v in state_dict.items()}
                
                # Load into eval model - check for missing/unexpected keys
                eval_state_dict = eval_model.state_dict()
                missing_keys = set(eval_state_dict.keys()) - set(state_dict.keys())
                unexpected_keys = set(state_dict.keys()) - set(eval_state_dict.keys())
                
                if self.eval_step < 3 and (missing_keys or unexpected_keys):
                    if missing_keys:
                        logger.warning(f"Missing keys in source state_dict: {list(missing_keys)[:5]}...")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys in source state_dict: {list(unexpected_keys)[:5]}...")
                
                eval_model.load_state_dict(state_dict, strict=False)
        
        logger.debug("Synced evaluation model weights from ParamMirror")

    def get_models(self) -> List[nn.Module]:
        """Get the list of models for evaluation.
        
        Returns our local evaluation models, synced from ParamMirror or disk.
        """
        if self.baseline:
            # Baseline mode: reload checkpoint if it has been updated
            self._load_baseline_checkpoint()
        else:
            # Distributed mode: sync weights from ParamMirror before returning
            self._sync_eval_model()
        return self._eval_models

    def _set_batchnorm_training_mode(self, model, training: bool):
        """Set BatchNorm layers to training/eval mode independently.
        
        When training=True, BatchNorm will use batch statistics instead of running stats.
        This is needed because:
        - Trainers run in train mode, so weights are calibrated for batch normalization
        - Running stats (running_mean, running_var) are often NOT synced in distributed training
        - Using default running_mean=0, running_var=1 with trained weights causes explosion
        
        Args:
            model: The model to modify
            training: Whether to set BatchNorm to training mode (use batch stats)
        """
        from torch.nn.modules.batchnorm import _BatchNorm
        
        bn_count = 0
        for module in model.modules():
            # Standard PyTorch BatchNorm variants
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.training = training
                bn_count += 1
            # SynchronizedBatchNorm and other _BatchNorm subclasses (used by BigGAN)
            elif isinstance(module, _BatchNorm):
                module.training = training
                bn_count += 1
            # BigGAN's custom ccbn and bn classes - they have stored_mean/stored_var buffers
            # and use F.batch_norm with self.training flag
            elif hasattr(module, 'stored_mean') and hasattr(module, 'stored_var'):
                module.training = training
                bn_count += 1
        
        if bn_count > 0:
            logger.debug(f"Set {bn_count} BatchNorm-like layers to training={training}")

    def _debug_model_stats(self, model, prefix=""):
        """Log statistics about model parameters for debugging."""
        total_params = 0
        nan_count = 0
        inf_count = 0
        param_means = []
        param_stds = []
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            nan_count += torch.isnan(param).sum().item()
            inf_count += torch.isinf(param).sum().item()
            param_means.append(param.float().mean().item())
            param_stds.append(param.float().std().item())
        
        mean_of_means = sum(param_means) / len(param_means) if param_means else 0
        mean_of_stds = sum(param_stds) / len(param_stds) if param_stds else 0
        
        logger.info(f"{prefix} Model stats: params={total_params}, NaN={nan_count}, Inf={inf_count}, "
                   f"mean={mean_of_means:.6f}, std={mean_of_stds:.6f}, "
                   f"device={next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")
        
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"{prefix} MODEL HAS {nan_count} NaN AND {inf_count} Inf VALUES!")
        
        # Debug BatchNorm running statistics (these are buffers, not parameters!)
        bn_issues = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    rm = module.running_mean
                    rv = module.running_var
                    rm_nan = torch.isnan(rm).any().item()
                    rv_nan = torch.isnan(rv).any().item()
                    rv_neg = (rv < 0).any().item()
                    rv_zero = (rv < 1e-10).any().item()
                    
                    if rm_nan or rv_nan or rv_neg or rv_zero:
                        bn_issues.append(f"{name}: rm_nan={rm_nan}, rv_nan={rv_nan}, rv_neg={rv_neg}, rv_near_zero={rv_zero}")
                    
                    # Log first BN layer stats
                    if len(bn_issues) == 0 and 'bn1' in name or 'layer1.0.bn1' in name:
                        logger.debug(f"{prefix} BatchNorm '{name}': running_mean=[{rm.min():.4f}, {rm.max():.4f}], "
                                    f"running_var=[{rv.min():.4f}, {rv.max():.4f}]")
        
        if bn_issues:
            logger.warning(f"{prefix} BatchNorm issues found: {bn_issues[:5]}")  # Show first 5

    @torch.no_grad()
    def forward_pass(self, inputs, labels=None):
        """
        Run forward pass through all pipeline stages.
        
        Returns:
            outputs: Model outputs
        """
        models = self.get_models()
        if not models:
            raise RuntimeError("No models available")
        
        # Debug: log model stats before forward pass
        if self.eval_step < 5:  # Only for first few evals to avoid spam
            self._debug_model_stats(models[0], prefix=f"[eval_step={self.eval_step}]")
        
        # Get model dtype from first parameter to ensure input dtype matches
        model_dtype = next(models[0].parameters()).dtype
        
        # Move inputs to device and cast to model dtype
        def to_device_and_dtype(t):
            if isinstance(t, torch.Tensor):
                t = t.to(self.device)
                # Only cast floating point tensors, not integer tensors (like labels)
                if t.is_floating_point():
                    t = t.to(dtype=model_dtype)
                return t
            return t
        
        if isinstance(inputs, torch.Tensor):
            x = to_device_and_dtype(inputs)
        elif isinstance(inputs, tuple):
            x = tuple(to_device_and_dtype(t) for t in inputs)
        else:
            x = inputs
        
        # Debug: log input stats
        if self.eval_step < 5:
            if isinstance(x, torch.Tensor):
                logger.debug(f"[eval_step={self.eval_step}] Input: shape={x.shape}, dtype={x.dtype}, "
                            f"mean={x.float().mean().item():.4f}, std={x.float().std().item():.4f}")
        
        if self.baseline:
            # Baseline mode: single BaselineModel that contains the full pipeline
            model = models[0]
            
            # For baseline, BatchNorm running stats are properly maintained during training
            # so we can use eval mode (default). But if running_stats seem uninitialized,
            # fall back to batch stats.
            if self.eval_step == 0 and not hasattr(self, '_bn_mode_logged'):
                logger.info("Baseline evaluation: using model's running stats for BatchNorm")
                self._bn_mode_logged = True
            
            task_type = self.cfg.data.task_type
            if task_type == "llm" or task_type == "node_pred":
                labels_on_device = labels.to(self.device) if labels is not None else None
                x = model(x, labels_on_device)
            else:
                x = model(x)
        else:
            # Distributed mode: run through pipeline stages
            # Note: We need to handle BatchNorm specially - if running_stats aren't synced from
            # trainers (which is common in distributed training), we need to compute batch stats.
            for model in models:
                # Set BatchNorm layers to training mode to use batch statistics
                # This is necessary because:
                # 1. Trainers run with model.train() - BatchNorm uses batch stats
                # 2. Weights are calibrated for batch normalization
                # 3. Running stats from trainers are often NOT synced (only params are)
                # 4. Default running_mean=0, running_var=1 causes output explosion
                if self.eval_step == 0 and not hasattr(self, '_bn_mode_logged'):
                    logger.info("Using batch statistics for BatchNorm layers (running stats not synced from trainers)")
                    self._bn_mode_logged = True
                self._set_batchnorm_training_mode(model, training=True)
                
                task_type = self.cfg.data.task_type
                if task_type == "llm" or task_type == "node_pred":
                    labels_on_device = labels.to(self.device) if labels is not None else None
                    x = model(x, labels_on_device)
                else:
                    x = model(x)
                
                # Restore BatchNorm to eval mode (don't update running stats)
                self._set_batchnorm_training_mode(model, training=False)
        
        # Debug: log output stats
        if self.eval_step < 5:
            if isinstance(x, torch.Tensor):
                has_nan = torch.isnan(x).any().item()
                has_inf = torch.isinf(x).any().item()
                logger.info(f"[eval_step={self.eval_step}] Output: shape={x.shape}, dtype={x.dtype}, "
                           f"mean={x.float().mean().item():.4f}, std={x.float().std().item():.4f}, "
                           f"min={x.float().min().item():.4f}, max={x.float().max().item():.4f}, "
                           f"NaN={has_nan}, Inf={has_inf}")
        
        return x

    def compute_metrics(self, outputs, labels, inputs=None) -> Dict[str, float]:
        """
        Compute task-specific metrics.
        
        Returns:
            Dict of metric_name -> value
        """
        task_type = self.cfg.data.task_type
        metrics = {}
        
        # Use different prefix for baseline vs distributed evaluation
        prefix = "baseline_eval" if self.baseline else "eval"
        
        if task_type == "cv":
            # Classification metrics
            loss = F.cross_entropy(outputs.float(), labels.to(outputs.device))
            metrics[f"{prefix}/loss"] = loss.item()
            
            # Accuracy
            preds = outputs.argmax(dim=-1)
            correct = (preds == labels.to(outputs.device)).float()
            metrics[f"{prefix}/accuracy"] = correct.mean().item()
            
            # Top-5 accuracy (if enough classes)
            if outputs.shape[-1] >= 5:
                _, top5_preds = outputs.topk(5, dim=-1)
                top5_correct = (top5_preds == labels.to(outputs.device).unsqueeze(-1)).any(dim=-1).float()
                metrics[f"{prefix}/top5_accuracy"] = top5_correct.mean().item()
                
        elif task_type == "llm":
            # Language model metrics
            loss = outputs.mean()
            metrics[f"{prefix}/loss"] = loss.item()
            metrics[f"{prefix}/perplexity"] = math.exp(min(loss.item(), 20))  # Cap to avoid overflow
            
        elif task_type == "speech":
            # Speech recognition metrics (CTC loss)
            from distqat.models.wav2vec2 import get_feat_extract_output_lengths
            from transformers import AutoConfig
            
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
            model_name = self.cfg.data.full_model_name
            config = AutoConfig.from_pretrained(model_name)
            input_lengths = get_feat_extract_output_lengths(
                attention_mask.sum(-1), config=config
            ).to(torch.long)
            
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            
            log_probs = F.log_softmax(outputs, dim=-1, dtype=torch.float32).transpose(0, 1)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs, flattened_targets, input_lengths, target_lengths,
                    reduction="mean"
                )
            metrics[f"{prefix}/loss"] = loss.item()
            
        elif task_type == "node_pred":
            # Graph node prediction
            loss = outputs.squeeze(0)[0] if outputs.dim() > 1 else outputs[0]
            metrics[f"{prefix}/loss"] = loss.item()
            
            # Accuracy from outputs if available (model may return (loss, logits))
            if outputs.dim() > 1 and outputs.shape[-1] > 1:
                logits = outputs.squeeze(0)[1:] if outputs.shape[0] == 1 else outputs[1:]
                # Note: node_pred typically returns loss directly, accuracy needs model-specific handling
                
        elif task_type == "image_gen":
            # Image generation metrics are computed via model.evaluate() in evaluate_image_gen()
            # This method handles the case where forward_pass was called (e.g., discriminator outputs)
            if outputs is not None and outputs.numel() > 1:
                metrics[f"{prefix}/D_output_mean"] = outputs[0].item()
                metrics[f"{prefix}/G_output_mean"] = outputs[-1].item()
            
        elif task_type == "rl":
            # RL metrics are computed during rollout collection
            # Here we just return what's in outputs
            if isinstance(outputs, tuple) and len(outputs) >= 4:
                _, newlogprob, entropy, newvalue = outputs
                metrics[f"{prefix}/entropy"] = entropy.mean().item()
                metrics[f"{prefix}/value_mean"] = newvalue.mean().item()
        
        return metrics

    @torch.no_grad()
    def evaluate_image_gen(self) -> Dict[str, float]:
        """
        Run evaluation for image generation models (BigGAN) using Inception Score and FID.
        
        Uses the model's built-in evaluate() method which:
        - Generates images using the generator
        - Computes IS (Inception Score) using a pretrained Inception network
        - Computes FID (Fréchet Inception Distance) using precomputed dataset moments
        
        Requires:
        - cfg.biggan['enable_eval'] = True
        - cfg.biggan['eval_moments_file'] pointing to precomputed dataset inception moments
        
        Returns:
            Dict of metrics including FID, IS_mean, IS_std, best_FID, best_IS
        """
        prefix = "baseline_eval" if self.baseline else "eval"
        metrics: Dict[str, float] = {}
        
        models = self.get_models()
        if not models:
            logger.warning("No models available for image_gen evaluation")
            return {f"{prefix}/error": 1.0}
        
        # BigGAN is typically a single-stage model
        if len(models) != 1:
            logger.warning(f"Expected 1 model for image_gen evaluation, got {len(models)}")
        
        model = models[0]
        
        # For BaselineModel, the BigGAN model is wrapped in model_pipeline
        if hasattr(model, 'model_pipeline') and len(model.model_pipeline) > 0:
            biggan = model.model_pipeline[0]
        else:
            biggan = model
        
        # Check if model has the evaluate method (BigGANAdapter)
        if not hasattr(biggan, 'evaluate'):
            logger.error("Model does not have evaluate() method for IS/FID calculation")
            return {f"{prefix}/error": 1.0, f"{prefix}/error_msg": "missing_evaluate_method"}
        
        # Check if evaluation is enabled in the model
        enable_eval = getattr(biggan, 'enable_eval', None)
        if enable_eval is False:
            logger.warning("Model evaluation is disabled (enable_eval=False). "
                          "Set cfg.biggan['enable_eval'] = True to enable IS/FID evaluation.")
            return {f"{prefix}/error": 1.0, f"{prefix}/error_msg": "eval_disabled"}
        
        logger.info(f"Running IS/FID evaluation at step {self.eval_step}...")
        
        try:
            # Move model to eval device if needed
            original_device = next(biggan.parameters()).device if hasattr(biggan, 'parameters') else None
            if original_device != self.device:
                biggan.to(self.device)
            
            # CRITICAL: Set BatchNorm layers to training mode (use batch statistics)
            # BigGAN uses custom BatchNorm classes (ccbn, bn, SynchronizedBatchNorm) with
            # stored_mean/stored_var buffers that are NOT properly synced in distributed training.
            # Using batch statistics instead produces correct normalization during generation.
            if self.eval_step == 0 and not hasattr(self, '_biggan_bn_mode_logged'):
                logger.info("BigGAN: Using batch statistics for BatchNorm layers (stored_* buffers not synced from trainers)")
                self._biggan_bn_mode_logged = True
            self._set_batchnorm_training_mode(biggan, training=True)
            
            # Call the model's built-in evaluate method
            eval_results = biggan.evaluate(self.eval_step)
            
            # Restore BatchNorm to eval mode
            self._set_batchnorm_training_mode(biggan, training=False)
            
            # Move back to original device
            if original_device is not None and original_device != self.device:
                biggan.to(original_device)
            
            if eval_results is None:
                logger.warning("Model evaluate() returned None - check inception network and moments file. "
                              "The eval_moments_file must exist and contain precomputed inception statistics.")
                return {f"{prefix}/error": 1.0, f"{prefix}/error_msg": "eval_returned_none"}
            
            # Extract metrics (convert numpy types to Python floats for JSON serialization)
            fid = eval_results.get("FID")
            is_mean = eval_results.get("IS_mean")
            is_std = eval_results.get("IS_std")
            best_fid = eval_results.get("best_FID")
            best_is = eval_results.get("best_IS")
            
            # Check for NaN metrics and log helpful diagnostics
            import math
            if fid is not None and (math.isnan(fid) or math.isinf(fid)):
                logger.warning(f"FID is {fid}! This usually indicates:")
                logger.warning("  1. Generator producing NaN/inf images (check BatchNorm stats)")
                logger.warning("  2. Covariance matrix is ill-conditioned (numerical instability)")
                logger.warning("  3. Too few inception images sampled")
                fid = None
            
            if is_mean is not None and (math.isnan(is_mean) or math.isinf(is_mean)):
                logger.warning(f"IS_mean is {is_mean}! Check generator output for NaN values.")
                is_mean = None
            
            # Track last valid metrics at evaluator level (backup for wandb continuity)
            if not hasattr(self, '_last_valid_fid'):
                self._last_valid_fid = None
            if not hasattr(self, '_last_valid_is'):
                self._last_valid_is = None
            
            # Update or carry forward FID
            if fid is not None:
                self._last_valid_fid = fid
                metrics[f"{prefix}/FID"] = float(fid)
            elif self._last_valid_fid is not None:
                # Carry forward last valid FID so wandb plot doesn't have gaps
                metrics[f"{prefix}/FID"] = float(self._last_valid_fid)
                logger.debug(f"FID unavailable, using last valid: {self._last_valid_fid:.4f}")
            
            # Update or carry forward IS
            if is_mean is not None:
                self._last_valid_is = is_mean
                metrics[f"{prefix}/IS_mean"] = float(is_mean)
            elif self._last_valid_is is not None:
                metrics[f"{prefix}/IS_mean"] = float(self._last_valid_is)
            
            if is_std is not None and not (math.isnan(is_std) or math.isinf(is_std)):
                metrics[f"{prefix}/IS_std"] = float(is_std)
            if best_fid is not None and not (math.isnan(best_fid) or math.isinf(best_fid)):
                metrics[f"{prefix}/best_FID"] = float(best_fid)
            if best_is is not None and not (math.isnan(best_is) or math.isinf(best_is)):
                metrics[f"{prefix}/best_IS"] = float(best_is)
            
            # Also set loss to FID for compatibility with other metrics (FID lower is better)
            if f"{prefix}/FID" in metrics:
                metrics[f"{prefix}/loss"] = metrics[f"{prefix}/FID"]
            
            logger.info(
                f"IS/FID evaluation complete: IS={is_mean:.3f}±{is_std:.3f}, FID={fid:.4f}"
                if is_mean is not None and is_std is not None and fid is not None
                else f"IS/FID evaluation complete: {eval_results}"
            )
            
        except Exception as e:
            logger.error(f"IS/FID evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            metrics[f"{prefix}/error"] = 1.0
            metrics[f"{prefix}/error_msg"] = str(e)[:100]  # Truncate for logging
        
        return metrics

    @torch.no_grad()
    def evaluate_epoch(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Run evaluation over the validation dataset.
        
        Args:
            max_batches: Maximum number of batches to evaluate (None = all)
            
        Returns:
            Dict of aggregated metrics
        """
        models = self.get_models()
        if not models:
            logger.warning("No models available from ParamMirror, skipping evaluation")
            return {}
        
        # Models are already on device and in eval mode (managed by get_models -> _sync_eval_model)
        for model in models:
            model.eval()
        
        all_metrics: Dict[str, List[float]] = {}
        num_samples = 0
        num_batches = 0
        
        task_type = self.cfg.data.task_type
        prefix = "baseline_eval" if self.baseline else "eval"
        
        # Skip RL task type - requires environment interaction
        if task_type == "rl":
            logger.info("RL evaluation requires environment interaction, skipping batch evaluation")
            return {f"{prefix}/note": "RL evaluation requires separate rollout collection"}
        
        # Image generation uses model's built-in IS/FID evaluation
        if task_type == "image_gen":
            logger.info("Image generation: running IS/FID evaluation via model.evaluate()")
            return self.evaluate_image_gen()
        
        logger.info("Starting evaluation epoch...")
        
        # Use persistent dataloader (trainer pattern - only recreate when exhausted)
        while True:
            try:
                result = self._fetch_batch_with_retry()
                if result is None:
                    logger.warning("No dataloader available, skipping evaluation")
                    break
                
                uid, batch = result
                inputs = batch["inputs"]
                labels = batch["labels"]
                
                # Handle different input types
                if isinstance(inputs, torch.Tensor):
                    batch_size = inputs.shape[0]
                elif isinstance(inputs, tuple):
                    batch_size = inputs[0].shape[0] if hasattr(inputs[0], 'shape') else 1
                else:
                    batch_size = 1
                
                # Forward pass
                outputs = self.forward_pass(inputs, labels)
                
                # Compute metrics
                batch_metrics = self.compute_metrics(outputs, labels, inputs)
                
                # Aggregate
                for key, value in batch_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
                
                num_samples += batch_size
                num_batches += 1
                
                if num_batches % 10 == 0:
                    logger.debug(f"Evaluated {num_batches} batches, {num_samples} samples")
                
                if max_batches is not None and num_batches >= max_batches:
                    break
                    
            except Exception as e:
                logger.warning(f"Error in evaluation batch: {e}")
                continue
        
        # Average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[key] = sum(values) / len(values) if values else 0.0
        
        prefix = "baseline_eval" if self.baseline else "eval"
        avg_metrics[f"{prefix}/num_samples"] = num_samples
        avg_metrics[f"{prefix}/num_batches"] = num_batches
        
        return avg_metrics

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to DHT for the monitor to collect and forward to wandb."""
        # Log to file/console
        mode_str = "baseline" if self.baseline else "distributed"
        logger.info(f"Evaluation ({mode_str}) step {step}:")
        logger.info(json.dumps(metrics, indent=2, sort_keys=True))
        
        # Publish to DHT for monitor to collect
        try:
            # Ensure eval/step is in the payload
            step_key = "baseline_eval/step" if self.baseline else "eval/step"
            if step_key not in metrics:
                metrics[step_key] = step
            
            self.dht.store(
                key=self._dht_key,
                subkey=self._dht_subkey,
                value=metrics,
                expiration_time=self._get_dht_time() + self._metrics_expiration,
                return_future=True,
            )
            logger.info(f"Published eval metrics to DHT key {self._dht_key}: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"Failed to publish eval metrics to DHT: {e}")

    def run(self, max_batches_per_eval: Optional[int] = 10):
        """
        Main evaluation loop. Runs periodically until interrupted.
        
        Args:
            max_batches_per_eval: Max batches per evaluation (None = full epoch)
        """
        mode_str = "baseline" if self.baseline else "distributed"
        logger.info(f"Starting {mode_str} evaluation loop with interval={self.eval_interval}s")
        logger.info(f"Max batches per evaluation: {max_batches_per_eval}")
        
        prefix = "baseline_eval" if self.baseline else "eval"
        
        while not self.stop_requested:
            try:
                # Run evaluation
                t0 = time.time()
                metrics = self.evaluate_epoch(max_batches=max_batches_per_eval)
                eval_time = time.time() - t0
                
                metrics[f"{prefix}/time_seconds"] = eval_time
                metrics[f"{prefix}/step"] = self.eval_step
                
                # Log metrics
                self.log_metrics(metrics, self.eval_step)
                self.eval_step += 1
                
                # Wait for next evaluation interval (interruptible via stop_event)
                sleep_time = max(0, self.eval_interval - eval_time)
                if sleep_time > 0:
                    logger.info(f"Sleeping {sleep_time:.1f}s until next evaluation...")
                    # Use event.wait() instead of time.sleep() so shutdown can interrupt it
                    if self.stop_event.wait(timeout=sleep_time):
                        logger.info("Sleep interrupted by shutdown signal")
                        break
                    
            except KeyboardInterrupt:
                logger.info("Evaluation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                time.sleep(self.eval_interval)
        
        logger.info("Evaluation loop finished")

    def shutdown(self):
        """Clean up resources."""
        self.stop_requested = True
        self.stop_event.set()  # Wake up any sleeping threads
        
        if self.param_mirror is not None:
            self.param_mirror.stop()
            self.param_mirror.join(timeout=5)
        
        try:
            if self.dht is not None:
                self.dht.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down DHT: {e}")
        
        logger.info("Evaluator shutdown complete")


def main(cfg: Config, eval_interval: int = 60, max_batches_per_eval: Optional[int] = 10, baseline: bool = False):
    """Main entry point for the evaluator."""
    evaluator = Evaluator(
        cfg=cfg,
        eval_interval=eval_interval,
        baseline=baseline,
    )
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        evaluator.stop_requested = True
        evaluator.stop_event.set()  # Wake up any sleeping threads
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    mode_str = "baseline" if baseline else "distributed"
    logger.info(f"Created Evaluator ({mode_str} mode) with config from {cfg}")
    logger.info(f"Eval interval: {eval_interval}s")
    
    try:
        evaluator.run(max_batches_per_eval=max_batches_per_eval)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error in evaluator: {e}")
        raise e
    finally:
        evaluator.shutdown()


if __name__ == "__main__":
    parse_args_with_extra_kwargs = click.option("--eval-interval", type=int, default=60)(parse_args)
    parse_args_with_extra_kwargs = click.option("--max-batches-per-eval", type=int, default=10)(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option("--baseline", is_flag=True, help="Evaluate baseline model from disk checkpoint instead of distributed peers")(parse_args_with_extra_kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*is used more than once. Remove its duplicate as parameters should be unique.*")
        res = parse_args_with_extra_kwargs(standalone_mode=False)
        if isinstance(res, int):
            quit()  # Help has been called
        elif isinstance(res, tuple):
            cfg, extra_kwargs = res
            main(cfg, **extra_kwargs)
        else:
            raise ValueError(f"Unexpected return type: {type(res)}")
