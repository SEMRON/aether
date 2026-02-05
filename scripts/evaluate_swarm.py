#!/usr/bin/env python3
"""
Lightweight evaluator for a running distqat swarm.

Key goals:
- Explicit CLI: you must pass --config-path and --network-initial-peers (JSON list or comma-separated)
- Minimal "magic": mirror parameters from peers, run eval, write metrics
- General across task types (cv/llm/speech/image_gen) using distqat's shared utilities

This intentionally does NOT:
- start/stop training processes
- guess peers from log files
- silently fall back to random weights if mirroring fails
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import click
import torch
from hivemind.dht import DHT
from hivemind.moe.server.layers import name_to_block
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from torch.utils.data import DataLoader

from distqat.config import Config, parse_args
from distqat.data import collate_fn, get_train_val_datasets
from distqat.models import kwargs_from_config
from distqat.optimizers import get_diloco_optimizer_cls_kwargs
from distqat.utils.compression import get_compression_kwargs
from distqat.utils.loss import task_type_loss

torch.multiprocessing.set_sharing_strategy("file_system")
use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _clone_module_tensors(module: torch.nn.Module) -> None:
    """Clone params/buffers to detach from shared-memory backing."""
    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone().detach()
        for b in module.buffers():
            b.data = b.data.clone().detach()
    gc.collect()


def _progress_summary(dht: DHT, prefix: str, stage_index: int) -> Dict[str, Any]:
    """
    Best-effort: read training progress from DHT and summarize.
    Returns an empty dict if not available.
    """
    key = f"{prefix}_{stage_index}_progress"
    try:
        resp = dht.get(key, latest=True)
        val = resp.value if hasattr(resp, "value") else resp
        if not isinstance(val, dict):
            return {}
        peers = []
        max_outer, max_inner = 0, 0
        for entry in val.values():
            if not hasattr(entry, "value") or entry.value is None:
                continue
            payload = entry.value
            if not isinstance(payload, dict):
                continue
            outer = int(payload.get("outer_step", 0) or 0)
            inner = int(payload.get("inner_step", 0) or 0)
            peers.append({"outer_step": outer, "inner_step": inner})
            max_outer = max(max_outer, outer)
            max_inner = max(max_inner, inner)
        return {
            "key": key,
            "num_peers": len(peers),
            "max_outer_step": max_outer,
            "max_inner_step": max_inner,
        }
    except Exception:
        return {}


def _run_pipeline_forward(models: Sequence[torch.nn.Module], x, labels=None):
    """
    Run forward pass through all pipeline stages.
    Labels are only passed to the last stage (matches distqat.distributed.model.BaselineModel).
    """
    num_stages = len(models)
    is_sequence = isinstance(x, (tuple, list))
    for idx, model in enumerate(models):
        is_last = idx == (num_stages - 1)
        if is_sequence:
            if labels is not None and is_last:
                x = model(*x, labels)
            else:
                x = model(*x)
            is_sequence = isinstance(x, (tuple, list))
        else:
            if labels is not None and is_last:
                x = model(x, labels)
            else:
                x = model(x)
    return x


def _evaluate_once(
    *,
    cfg: Config,
    models: List[torch.nn.Module],
    dev: torch.device,
    split: str,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    eval_step: int,
) -> Dict[str, Any]:
    """
    Evaluate current mirrored models and return metrics.
    For image_gen, calls BigGANAdapter.evaluate() (no dataset iteration).
    For other tasks, runs a small loader and returns task-specific metrics.
    """
    metrics: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "task_type": cfg.data.task_type,
        "device": str(dev),
        "hostname": os.uname().nodename,
        "split": split,
        "eval_step": int(eval_step),
    }

    # Move models to device for eval
    for m in models:
        m.to(dev)
        m.eval()

    try:
        if cfg.data.task_type == "image_gen":
            if len(models) != 1:
                raise RuntimeError(f"Expected 1 stage for image_gen, got {len(models)}")
            biggan = models[0]
            if not hasattr(biggan, "evaluate"):
                raise RuntimeError("Model does not implement evaluate() for image_gen")
            res = biggan.evaluate(eval_step)
            if res is None:
                enable_eval = getattr(biggan, "enable_eval", None)
                moments = None
                try:
                    moments = getattr(biggan, "config", {}).get("eval_moments_file")
                except Exception:
                    moments = None
                raise RuntimeError(
                    "BigGAN evaluate() returned None. Usually evaluation is disabled "
                    f"(enable_eval={enable_eval}) or inception moments are missing/invalid "
                    f"(eval_moments_file={moments!r})."
                )
            metrics.update(
                {
                    "IS_mean": float(res.get("IS_mean")),
                    "IS_std": float(res.get("IS_std")),
                    "FID": float(res.get("FID")),
                    "best_IS": float(res.get("best_IS")),
                    "best_FID": float(res.get("best_FID")),
                }
            )
            return metrics

        train_ds, val_ds = get_train_val_datasets(cfg.data)
        ds = train_ds if split == "train" else val_ds
        cfn = collate_fn(cfg.data, cfg.model_pipeline.pipeline[0])
        loader = DataLoader(
            ds,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            collate_fn=cfn,
        )

        amp_dtype = None
        if dev.type == "cuda":
            if cfg.data.precision == "fp16-mixed":
                amp_dtype = torch.float16
            elif cfg.data.precision == "bf16-mixed":
                amp_dtype = torch.bfloat16

        total_loss = 0.0
        total_items = 0
        total_correct = 0
        total_target_len = 0
        total_input_len = 0

        with torch.inference_mode():
            for i, (_uids, batch) in enumerate(loader):
                if i >= int(max_batches):
                    break
                inputs = batch["inputs"]
                labels = batch["labels"]
                if isinstance(inputs, tuple):
                    inputs = tuple(x.to(dev) if hasattr(x, "to") else x for x in inputs)
                else:
                    inputs = inputs.to(dev)
                labels = labels.to(dev) if hasattr(labels, "to") else labels

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else torch.no_grad()
                )
                with autocast_ctx:
                    outputs = _run_pipeline_forward(models, inputs, labels)
                    loss_t = task_type_loss(cfg, inputs, outputs, labels)

                bsz = int(labels.shape[0]) if hasattr(labels, "shape") else int(batch_size)
                total_loss += float(loss_t.item()) * bsz
                total_items += bsz

                if cfg.data.task_type in ("cv", "node_pred"):
                    # Expect classification logits
                    preds = outputs.argmax(dim=-1)
                    y = labels
                    # Allow labels to be shaped (B, 1) or similar; compare on last dim.
                    if hasattr(y, "ndim") and y.ndim > 1:
                        y = y.view(-1)
                    if hasattr(preds, "ndim") and preds.ndim > 1:
                        preds = preds.view(-1)
                    total_correct += int((preds == y).sum().item())

                if cfg.data.task_type == "speech":
                    # We don't decode WER/CER here (would add heavy deps and complexity).
                    # Provide lightweight length statistics for sanity checking.
                    try:
                        if isinstance(labels, torch.Tensor):
                            # common convention: pad labels with -100 or -1
                            mask = labels >= 0
                            total_target_len += int(mask.sum().item())
                        if isinstance(inputs, torch.Tensor) and inputs.ndim >= 2:
                            # (B, T) or (B, C, T)
                            total_input_len += int(inputs.shape[-1]) * int(bsz)
                    except Exception:
                        pass

        if total_items == 0:
            raise RuntimeError("No batches evaluated (empty dataset or max_batches=0?)")

        mean_loss = total_loss / total_items
        metrics.update(
            {
                "loss": float(mean_loss),
                "num_items": int(total_items),
                "num_batches": int(min(int(max_batches), math.ceil(total_items / int(batch_size)))),
            }
        )

        # Add task-specific metrics (keep these small/cheap to compute).
        if cfg.data.task_type == "cv":
            metrics["accuracy_top1"] = float(total_correct / total_items)
        elif cfg.data.task_type == "node_pred":
            metrics["accuracy_top1"] = float(total_correct / total_items)
        elif cfg.data.task_type == "llm":
            metrics["perplexity"] = float(math.exp(mean_loss)) if mean_loss < 50 else float("inf")
        elif cfg.data.task_type == "speech":
            if total_items > 0:
                metrics["avg_input_len"] = float(total_input_len / total_items) if total_input_len > 0 else None
            if total_target_len > 0:
                metrics["avg_target_len"] = float(total_target_len / total_items)
        elif cfg.data.task_type in ("rl",):
            # RL uses a different evaluation loop (env rollouts). Don't pretend this works.
            raise RuntimeError("RL evaluation is not supported by evaluate_swarm.py (use a rollout-based evaluator).")
        else:
            # Unknown or unsupported task type
            raise RuntimeError(f"Unsupported task_type={cfg.data.task_type!r} for evaluation.")
        return metrics
    finally:
        # Move models back to CPU and reclaim GPU memory between evals
        for m in models:
            m.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SwarmParamMirror:
    def __init__(
        self,
        *,
        cfg: Config,
        dht: DHT,
        stage_indices: Optional[Sequence[int]] = None,
        load_state_timeout: float,
    ):
        self.cfg = cfg
        self.dht = dht
        self.stage_indices = set(stage_indices) if stage_indices is not None else None
        self.load_state_timeout = float(load_state_timeout)

        self.models: List[torch.nn.Module] = []
        self.optimizers: List[object] = []
        self.stage_ids: List[int] = []

        compression = get_compression_kwargs(cfg.network.hivemind_compression)

        for stage_index, pipeline_step_cfg in enumerate(cfg.model_pipeline.pipeline):
            if self.stage_indices is not None and stage_index not in self.stage_indices:
                continue

            model_name, stage = pipeline_step_cfg.model_name.split(".")
            expert_cls = f"{model_name}.{stage}"
            block_ctor = name_to_block.get(expert_cls)
            if block_ctor is None:
                raise ValueError(f"Unknown expert class {expert_cls!r}. Is it registered in name_to_block?")

            aliases = None
            if cfg.data.task_type == "image_gen":
                # BigGANAdapter expects kwarg name "config" for the BigGAN dict.
                aliases = {"config": cfg.biggan}

            model_kwargs = kwargs_from_config(block_ctor.__init__, pipeline_step_cfg, cfg.data, aliases=aliases)
            model = block_ctor(**model_kwargs)
            model.to("cpu")
            model.eval()

            run_id = f"{cfg.experiment_prefix}_{stage_index}"
            optim_cls, optim_kwargs = get_diloco_optimizer_cls_kwargs(run_id, cfg.diloco, compression)
            optim_kwargs["load_state_timeout"] = self.load_state_timeout
            if cfg.data.task_type == "image_gen":
                optim_kwargs["expert"] = model

            optimizer = optim_cls(
                params=model.parameters(),
                avg_only_params=[],
                dht=dht,
                expert=model,
                **optim_kwargs,
            )

            self.models.append(model)
            self.optimizers.append(optimizer)
            self.stage_ids.append(stage_index)

        if not self.models:
            raise RuntimeError("No stages selected for evaluation (empty pipeline or stage filter too strict)")

    def refresh(self) -> Dict[int, int]:
        """Mirror state from peers. Returns dict {stage_index -> outer_step}."""
        outer_steps: Dict[int, int] = {}
        for stage_index, model, optimizer in zip(self.stage_ids, self.models, self.optimizers):
            prog = _progress_summary(self.dht, self.cfg.experiment_prefix, stage_index)
            if prog:
                logger.info(
                    "Progress %s: peers=%s max_outer=%s max_inner=%s",
                    prog.get("key"),
                    prog.get("num_peers"),
                    prog.get("max_outer_step"),
                    prog.get("max_inner_step"),
                )

            # Avoid the optimizer's internal retry loop by calling averager directly when present.
            if hasattr(optimizer, "averager"):
                optimizer.averager.load_state_from_peers(timeout=self.load_state_timeout)
            else:
                optimizer.load_state_from_peers()

            _clone_module_tensors(model)
            outer_steps[stage_index] = int(getattr(optimizer, "outer_step", 0) or 0)

        return outer_steps

    def shutdown(self) -> None:
        for opt in self.optimizers:
            try:
                if hasattr(opt, "shutdown"):
                    opt.shutdown()
            except Exception:
                pass


def _parse_stage_indices(spec: Optional[str]) -> Optional[List[int]]:
    """
    Parse "0,1,2" or "0" into [0, 1, 2]. Returns None for "all"/None.
    """
    if spec is None:
        return None
    s = spec.strip().lower()
    if s in ("", "all"):
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts]


def main(
    cfg: Config,
    *,
    eval_interval: float,
    one_shot: bool,
    split: str,
    batch_size: Optional[int],
    max_batches: int,
    num_workers: int,
    out_dir: Optional[str],
    save_checkpoint_dir: Optional[str],
    stages: Optional[str],
    load_state_timeout: float,
):
    if not cfg.network.initial_peers:
        raise RuntimeError("No initial peers configured. Pass --network-initial-peers explicitly.")

    dev = _resolve_device(cfg.device)

    # Make BigGAN evaluation work out of the box.
    if cfg.data.task_type == "image_gen" and cfg.biggan is not None:
        cfg.biggan["enable_eval"] = True
        cfg.biggan["device"] = str(dev)

    logger.info("Evaluator starting")
    logger.info("  config.experiment_prefix=%s", cfg.experiment_prefix)
    logger.info("  initial_peers=%s", cfg.network.initial_peers)
    logger.info("  device=%s", dev)
    logger.info("  task_type=%s", cfg.data.task_type)

    dht = DHT(
        start=True,
        initial_peers=cfg.network.initial_peers,
        host_maddrs=cfg.network.host_maddrs,
        announce_maddrs=cfg.network.announce_maddrs,
        client_mode=True,
    )

    stage_indices = _parse_stage_indices(stages)
    mirror = SwarmParamMirror(
        cfg=cfg,
        dht=dht,
        stage_indices=stage_indices,
        load_state_timeout=float(load_state_timeout),
    )

    out_path = None
    if out_dir is not None:
        out_path = Path(out_dir) / "metrics.jsonl"

    ckpt_dir = Path(save_checkpoint_dir) if save_checkpoint_dir is not None else None

    try:
        while True:
            outer_steps = mirror.refresh()
            stage0_step = int(outer_steps.get(min(outer_steps.keys()), 0) or 0)

            metrics = _evaluate_once(
                cfg=cfg,
                models=mirror.models,
                dev=dev,
                split=split,
                batch_size=int(batch_size or cfg.diloco.batch_size_per_step),
                max_batches=int(max_batches),
                num_workers=int(num_workers),
                eval_step=stage0_step,
            )
            metrics["mirrored_outer_steps"] = outer_steps

            logger.info("Metrics: %s", json.dumps(metrics, sort_keys=True))

            if out_path is not None:
                _write_jsonl(out_path, [metrics])

            if ckpt_dir is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                combined = {}
                for stage_idx, model in zip(mirror.stage_ids, mirror.models):
                    for k, v in model.state_dict().items():
                        combined[f"model_pipeline.{stage_idx}.{k}"] = v
                tmp = ckpt_dir / ".checkpoint_last.pt.tmp"
                dst = ckpt_dir / "checkpoint_last.pt"
                torch.save(combined, tmp)
                os.replace(tmp, dst)

            if one_shot:
                break
            time.sleep(float(eval_interval))
    finally:
        try:
            mirror.shutdown()
        except Exception:
            pass
        try:
            dht.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    parse_args_with_extra_kwargs = click.option("--eval-interval", type=float, default=60.0, show_default=True)(
        parse_args
    )
    parse_args_with_extra_kwargs = click.option("--one-shot", is_flag=True, default=False)(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option(
        "--split",
        type=click.Choice(["train", "validation"]),
        default="validation",
        show_default=True,
    )(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for non-image tasks (defaults to cfg.diloco.batch_size_per_step).",
    )(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option(
        "--max-batches",
        type=int,
        default=20,
        show_default=True,
        help="Max batches for non-image tasks.",
    )(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option(
        "--num-workers",
        type=int,
        default=0,
        show_default=True,
    )(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option(
        "--out-dir",
        type=str,
        default=None,
        help="If set, append JSONL metrics to <out-dir>/metrics.jsonl.",
    )(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option(
        "--save-checkpoint-dir",
        type=str,
        default=None,
        help="If set, write a BaselineModel-compatible checkpoint_last.pt into this directory.",
    )(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option(
        "--stages",
        type=str,
        default=None,
        help='Stages to mirror/evaluate, e.g. "0" or "0,1". Default: all stages.',
    )(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option(
        "--load-state-timeout",
        type=float,
        default=300.0,
        show_default=True,
        help="Timeout (seconds) for state mirroring from peers per stage.",
    )(parse_args_with_extra_kwargs)

    with click.Context(parse_args_with_extra_kwargs):
        res = parse_args_with_extra_kwargs(standalone_mode=False)
        if isinstance(res, int):
            raise SystemExit(res)
        if not isinstance(res, tuple) or len(res) != 2:
            raise RuntimeError(f"Unexpected parse_args return: {type(res)} {res}")
        cfg, extra_kwargs = res
        main(cfg, **extra_kwargs)

