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
        stats = state_dict.get("stats", {})
        self.update_count = stats.get("updates", 0)
        self.examples_processed = stats.get("examples_processed", 0)
        self.model.load_state_dict(state_dict["model"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])


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
            
            model_kwargs = kwargs_from_config(block_ctor.__init__, pipeline_step_cfg, cfg.data)
            model = block_ctor(**model_kwargs)
            model.to("cpu")
            run_id = f"{cfg.experiment_prefix}_{stage_index}"
            optim_cls, optim_kwargs = get_diloco_optimizer_cls_kwargs(run_id, cfg.diloco)
            try:
                optimizer = optim_cls(
                    params=model.parameters(),
                    avg_only_params=[],
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
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if is_directory(checkpoint_dir):
                load_experts(self._expert_backends, checkpoint_dir)
            if len(self._expert_backends) > 0:
                try:
                    self._checkpoint_saver = CheckpointSaver(
                        self._expert_backends,
                        checkpoint_dir,
                        self.refresh_every,
                        keep_history=cfg.checkpoint_keep_history,
                    )
                    self._checkpoint_saver.start()
                except Exception as e:
                    logger.warning(f"ParamMirror: failed to start CheckpointSaver: {e}")

    def run(self):
        while not self.stop_evt.wait(self.refresh_every):
            for idx, (model, optimizer, run_id, expert_uid) in enumerate(self._mirrors):
                optimizer.load_state_from_peers()

    def stop(self):
        self.stop_evt.set()
        if self._checkpoint_saver is not None:
            self._checkpoint_saver.stop.set()
            self._checkpoint_saver.join()
