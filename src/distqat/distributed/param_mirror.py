from typing import List, Optional, Dict, Tuple
import threading
from pathlib import Path
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.dht import DHT
from hivemind.moe.server.layers import name_to_block

from distqat.config import Config
from distqat.distributed.server.checkpoints import (
    CheckpointSaver,
    is_directory,
    load_experts,
)
from distqat.optimizers import get_diloco_optimizer_cls_kwargs
from distqat.models import kwargs_from_config
from distqat.utils.compression import get_compression_kwargs
from distqat.attach import attach_quantizers

logger = get_logger(__name__)
use_hivemind_log_handler("in_root_logger")


class _MirrorBackend:
    """
    Minimal adapter providing get_full_state/load_full_state so we can reuse
    server-side checkpointing utilities for the client-side parameter mirror.
    """

    def __init__(self, name: str, model: object, optimizer: object):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.update_count = 0
        self.examples_processed = 0

    def get_full_state(self) -> Dict:
        return {
            "stats": {
                "updates": self.update_count,
                "examples_processed": self.examples_processed,
            },
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_full_state(self, state_dict: Dict):
        """
        Load checkpoint state.

        Supports:
        - distributed expert checkpoints: {"model": ..., "optimizer": ..., "stats": ...}
        - baseline checkpoints: raw model state_dict (as saved by distqat.utils.baseline_checkpoints)
        """
        optimizer_state = None
        if isinstance(state_dict, dict) and "model" in state_dict:
            stats = state_dict.get("stats", {}) or {}
            self.update_count = stats.get("updates", 0)
            self.examples_processed = stats.get("examples_processed", 0)
            model_state = state_dict["model"]
            optimizer_state = state_dict.get("optimizer")
        else:
            # Baseline format: raw model weights only.
            model_state = state_dict

        # Baseline models may wrap stages under "model_pipeline.0." prefix; strip if present.
        if isinstance(model_state, dict) and any(k.startswith("model_pipeline.0.") for k in model_state.keys()):
            logger.info(f"ParamMirror: stripping 'model_pipeline.0.' prefix for {self.name}")
            model_state = {
                k[len("model_pipeline.0."):] if k.startswith("model_pipeline.0.") else k: v
                for k, v in model_state.items()
            }

        # Be permissive: checkpoints may include extra keys (e.g., from different partitioning / quantization).
        self.model.load_state_dict(model_state, strict=False)

        # Optimizer restore is best-effort.
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except ValueError as e:
                logger.warning(
                    "ParamMirror: failed to restore optimizer state for %s (continuing with fresh optimizer): %r",
                    self.name,
                    e,
                    exc_info=True,
                )


class ParamMirror(threading.Thread):
    def __init__(self, cfg: Config, dht: DHT, *, refresh_every: int = 30):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.dht = dht
        self.refresh_every = refresh_every
        self.stop_evt = threading.Event()

        self._mirrors: List[Tuple[object, object, str, str]] = []
        self._expert_backends: Dict[str, _MirrorBackend] = {}
        self._checkpoint_saver: Optional[CheckpointSaver] = None

        models: List[str] = []
        stages: List[str] = []
        for pipeline_step_cfg in cfg.model_pipeline.pipeline:
            model, stage = pipeline_step_cfg.model_name.split(".")
            models.append(model)
            stages.append(stage)

        for stage_index, pipeline_step_cfg in enumerate(cfg.model_pipeline.pipeline):
            expert_cls = f"{models[stage_index]}.{stages[stage_index]}"
            expert_uid = f"{stages[stage_index]}.0.param_mirror.0"
            block_ctor = name_to_block.get(expert_cls)
            if block_ctor is None:
                logger.warning(f"ParamMirror: unknown expert_cls {expert_cls}, skipping stage {stage_index}")
                continue
            
            aliases = {"config": pipeline_step_cfg.extra} if len(pipeline_step_cfg.extra.keys()) > 0 else None
            model_kwargs = kwargs_from_config(block_ctor.__init__, pipeline_step_cfg, cfg.data, aliases=aliases)
            model = block_ctor(**model_kwargs)
            
            # Attach quantizers if quantization is enabled (must match trainer model structure)
            avg_only_params = []
            if not getattr(cfg, 'disable_quant', True) and hasattr(cfg, 'quant') and cfg.quant is not None:
                logger.info(f"ParamMirror: attaching quantizers to stage {stage_index}")
                model, avg_only_params = attach_quantizers(model, cfg.quant)
            
            model.to("cpu")
            run_id = f"{cfg.experiment_prefix}_{stage_index}"
            compression = get_compression_kwargs(cfg.network.hivemind_compression)
            optim_cls, optim_kwargs = get_diloco_optimizer_cls_kwargs(run_id, cfg.diloco, compression)
            if cfg.data.task_type == "image_gen":
                optim_kwargs["expert"] = model
            try:
                optimizer = optim_cls(
                    params=model.parameters(),
                    avg_only_params=avg_only_params,
                    dht=self.dht,
                    **optim_kwargs,
                )
            except Exception as e:
                logger.warning(f"ParamMirror: failed to create optimizer for stage {stage_index}: {e}")
                continue
            try:
                optimizer.load_state_from_peers()
            except Exception:
                pass
            self._mirrors.append((model, optimizer, run_id, expert_uid))
            self._expert_backends[expert_uid] = _MirrorBackend(expert_uid, model, optimizer)

        checkpoint_dir = cfg.checkpoint_dir if cfg.checkpoint_dir else None
        logger.info(f"ParamMirror: checkpoint_dir: {checkpoint_dir}")
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if is_directory(checkpoint_dir):
                load_experts(self._expert_backends, checkpoint_dir)
                logger.info(f"ParamMirror: loaded experts from checkpoint_dir: {checkpoint_dir}")
            if len(self._expert_backends) > 0:
                try:
                    self._checkpoint_saver = CheckpointSaver(
                        self._expert_backends,
                        checkpoint_dir,
                        self.refresh_every,
                        keep_history=cfg.checkpoint_keep_history,
                    )
                    logger.info(f"ParamMirror: starting CheckpointSaver")
                    self._checkpoint_saver.start()
                except Exception as e:
                    logger.warning(f"ParamMirror: failed to start CheckpointSaver: {e}")

    def run(self):
        while not self.stop_evt.wait(self.refresh_every):
            for idx, (model, optimizer, run_id, expert_uid) in enumerate(self._mirrors):
                optimizer.load_state_from_peers()

    def get_all_models(self):
        return [model for model, _, _, _ in self._mirrors]

    def stop(self):
        self.stop_evt.set()
        if self._checkpoint_saver is not None:
            self._checkpoint_saver.stop.set()
            self._checkpoint_saver.join()
