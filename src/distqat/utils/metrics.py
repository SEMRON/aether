from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, StrictFloat, conint
from hivemind.dht import DHT
from hivemind.utils import get_dht_time
from hivemind.utils.logging import get_logger
import torch

logger = get_logger(__name__)

class LocalMetrics(BaseModel):
    step: conint(ge=0, strict=True)
    trainer_id: conint(ge=-2, strict=True)
    # These are logged by `on_step_end`. They are optional so other per-step scalars can be published
    # without clobbering or requiring loss/sps to exist first.
    loss: Optional[StrictFloat] = None
    sps: Optional[StrictFloat] = None
    # Arbitrary scalar metrics (e.g., "charts/episodic_return") that the monitor can forward to wandb.
    scalars: Dict[str, StrictFloat] = Field(default_factory=dict)


class MetricsLogger:
    """
    This callback logs metrics to the DHT for the monitor to collect.
    No Wandb initialization here - that's handled centrally by the monitor.
    """

    def __init__(
            self,
            dht: DHT,
            model: torch.nn.Module,
            local_public_key: Union[bytes, str],
            experiment_prefix: str,
            statistics_expiration: float,
            trainer_id: int,
            key_suffix: str = "_metrics",
        ):
            super().__init__()
            self.model = model
            self.dht = dht
            self.local_public_key = local_public_key
            self.experiment_prefix = experiment_prefix
            self.statistics_expiration = statistics_expiration
            self.last_reported_collaboration_step = -1
            self.trainer_id = trainer_id
            self._latest_metrics_by_step: Dict[int, Dict[str, Any]] = {}
            self.key_suffix = str(key_suffix)
            logger.debug(f"CollaborativeCallback initialized")

    def _store_metrics(self, payload: Dict[str, Any]) -> None:
        """Store a single merged metrics payload under this trainer's subkey."""
        self.dht.store(
            key=self.experiment_prefix + self.key_suffix,
            subkey=self.local_public_key,
            value=payload,
            expiration_time=get_dht_time() + self.statistics_expiration,
            return_future=True,
        )

    def _merge_and_store(self, step: int, patch: Dict[str, Any]) -> None:
        """
        Merge a patch into the latest known metrics for `step` and store the merged payload.

        Important: DHT values are stored per-subkey. If we store episodic metrics separately from loss,
        we'd overwrite the existing value. This merge keeps them together.
        """
        base = self._latest_metrics_by_step.get(step, {})
        merged = dict(base)

        # Merge scalar sub-dict.
        scalars = dict(merged.get("scalars") or {})
        patch_scalars = patch.get("scalars") or {}
        if isinstance(patch_scalars, dict):
            scalars.update(patch_scalars)
        merged.update(patch)
        merged["scalars"] = scalars

        # Ensure required identity fields are always present.
        merged["step"] = step
        merged["trainer_id"] = self.trainer_id

        self._latest_metrics_by_step[step] = merged
        # Prevent unbounded growth if training runs for a long time.
        if len(self._latest_metrics_by_step) > 256:
            oldest_step = min(self._latest_metrics_by_step.keys())
            self._latest_metrics_by_step.pop(oldest_step, None)
        self._store_metrics(merged)

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            x = float(value)
        except Exception:
            return None
        if not math.isfinite(x):
            return None
        return x

    def on_step_end(self, global_step, loss, sps=0.0):
        if global_step != self.last_reported_collaboration_step:
            self.last_reported_collaboration_step = global_step

            try:
                # Merge into existing metrics for this step (if any scalars were logged first).
                patch = LocalMetrics(
                    step=int(global_step),
                    trainer_id=self.trainer_id,
                    loss=float(loss),
                    sps=float(sps),
                ).model_dump()
                self._merge_and_store(int(global_step), patch)
                logger.debug(f"Metrics stored to DHT: {patch}")
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                logger.warning(f"Failed to store metrics to DHT: {e}")

    
    def log_scalar(self, name: str, value: Any, step: int) -> None:
        """
        Record a scalar metric for the monitor to pick up and forward to wandb.

        This intentionally does NOT touch wandb directly (the monitor owns wandb).
        """
        step = int(step)
        x = self._to_float(value)
        if x is None:
            return
        try:
            self._merge_and_store(step, {"scalars": {str(name): float(x)}})
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.debug(f"Failed to store scalar '{name}' to DHT: {e}")

    @staticmethod
    def _extract_episode_pairs_from_infos(infos: Any) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Extract (return, length) pairs from Gymnasium/Gym vector env info dictionaries.

        Supports:
        - infos["final_info"] = list[dict] where dict may contain {"episode": {...}}
        - infos["episode"] = {"r": np.ndarray|float, "l": np.ndarray|int, ...}
        """
        if not isinstance(infos, dict):
            return []

        pairs: List[Tuple[Optional[float], Optional[float]]] = []

        # Newer vector API: per-env final_info list
        final_info = infos.get("final_info")
        if isinstance(final_info, (list, tuple)):
            for fi in final_info:
                if not (fi and isinstance(fi, dict)):
                    continue
                ep = fi.get("episode")
                if not isinstance(ep, dict):
                    continue
                pairs.extend(MetricsLogger._extract_episode_pairs_from_episode_dict(ep))

        # Older/alternate vector API: episode directly on infos
        ep = infos.get("episode")
        if isinstance(ep, dict):
            pairs.extend(MetricsLogger._extract_episode_pairs_from_episode_dict(ep))

        return pairs

    @staticmethod
    def _extract_episode_pairs_from_episode_dict(ep: Dict[str, Any]) -> List[Tuple[Optional[float], Optional[float]]]:
        r = ep.get("r")
        l = ep.get("l")
        pairs: List[Tuple[Optional[float], Optional[float]]] = []

        # r/l may be scalar or per-env arrays/lists
        if isinstance(r, (list, tuple, np.ndarray)):
            for i, ri in enumerate(r):
                li = None
                if isinstance(l, (list, tuple, np.ndarray)) and i < len(l):
                    li = l[i]
                pairs.append((MetricsLogger._to_float(ri), MetricsLogger._to_float(li)))
        else:
            pairs.append((MetricsLogger._to_float(r), MetricsLogger._to_float(l)))
        return pairs

    def log_episodic_from_infos(self, infos: Any, step: int) -> None:
        """
        Log episodic return/length from an env `infos` dict.

        We aggregate multiple finished episodes from a vector env into a single mean value per step,
        then publish via `log_scalar` so the monitor can forward to wandb.
        """
        pairs = self._extract_episode_pairs_from_infos(infos)
        if not pairs:
            return

        returns = [r for r, _ in pairs if r is not None]
        lengths = [l for _, l in pairs if l is not None]
        if returns:
            mean_r = float(np.mean(np.asarray(returns, dtype=np.float64)))
            logger.info(f"env_step={step}, episodic_return_mean={mean_r}")
            self.log_scalar("charts/episodic_return", mean_r, step)
        if lengths:
            mean_l = float(np.mean(np.asarray(lengths, dtype=np.float64)))
            self.log_scalar("charts/episodic_length", mean_l, step)