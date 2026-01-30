#!/usr/bin/env python3

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import click
import torch
import torch.nn.functional as F
from deepmerge import always_merger
from pydanclick import from_pydantic
from pydantic_yaml import parse_yaml_file_as
from torch.utils.data import DataLoader
from transformers import AutoConfig

import visualize_eval_sweep

from distqat.attach import attach_quantizers
from distqat.config import Config
from distqat.data import (
    collate_fn,
    get_train_val_datasets,
)
from distqat.distributed.model import BaselineModel
from distqat.models.wav2vec2 import get_feat_extract_output_lengths


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: Path) -> None:
    obj = torch.load(checkpoint_path, map_location="cpu")
    # Baseline checkpoints store a raw state_dict; distributed checkpoints store {"model": ..., ...}
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state_dict = obj["model"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(f"Unexpected checkpoint payload type: {type(obj)}")

    def _try_load(sd: Dict[str, Any]) -> tuple[list[str], list[str]]:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return list(missing), list(unexpected)

    missing, unexpected = _try_load(state_dict)

    # Common fixup: some checkpoints are saved from DDP-wrapped modules with "module." prefixes.
    if (missing and unexpected) and any(k.startswith("module.") for k in state_dict.keys()):
        stripped_module = {
            (k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        missing, unexpected = _try_load(stripped_module)

    # Common fixup: distributed expert checkpoints often store a *bare expert* state_dict
    # (e.g. "G.blocks.0.weight") while BaselineModel expects "model_pipeline.0.<...>".
    if missing and unexpected:
        model_keys = list(model.state_dict().keys())
        model_expects_pipeline = any(k.startswith("model_pipeline.0.") for k in model_keys)
        ckpt_has_pipeline = any(k.startswith("model_pipeline.") for k in state_dict.keys())
        if model_expects_pipeline and not ckpt_has_pipeline:
            remapped = {f"model_pipeline.0.{k}": v for k, v in state_dict.items()}
            missing, unexpected = _try_load(remapped)

    # Heuristic fixup: distributed expert checkpoints typically save the expert module state_dict
    # (e.g. "resnet.conv1.weight") while BaselineModel expects it nested under "model_pipeline.0."
    if (missing and unexpected) and any(k.startswith("resnet.") for k in state_dict.keys()):
        model_keys = list(model.state_dict().keys())
        if any(k.startswith("model_pipeline.0.") for k in model_keys) and not any(
            k.startswith("model_pipeline.") for k in state_dict.keys()
        ):
            remapped = {f"model_pipeline.0.{k}": v for k, v in state_dict.items()}
            missing, unexpected = _try_load(remapped)

    # Inverse fixup: checkpoint has model_pipeline.* keys but model is a bare expert module.
    if (missing and unexpected) and any(k.startswith("model_pipeline.") for k in state_dict.keys()):
        model_keys = list(model.state_dict().keys())
        if not any(k.startswith("model_pipeline.") for k in model_keys):
            stripped = {}
            for k, v in state_dict.items():
                if k.startswith("model_pipeline.0."):
                    stripped[k[len("model_pipeline.0.") :]] = v
                else:
                    stripped[k] = v
            missing, unexpected = _try_load(stripped)

    if missing:
        print(
            f"WARNING: Missing keys when loading checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if unexpected:
        print(
            f"WARNING: Unexpected keys when loading checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )


def _extract_checkpoint_stats(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Best-effort extraction of training progress from a checkpoint payload.

    Supported:
    - extended baseline checkpoints: {"stats": {"updates": ..., "examples_processed": ...}, "model": {...}, ...}
    - legacy raw state_dict checkpoints: no stats available
    """
    try:
        obj = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        return {"checkpoint_read_error": str(e)}

    if isinstance(obj, dict):
        stats = obj.get("stats")
        if isinstance(stats, dict):
            out: Dict[str, Any] = {}
            if "updates" in stats:
                out["updates"] = stats.get("updates")
            if "examples_processed" in stats:
                out["examples_processed"] = stats.get("examples_processed")
            return out
    return {}


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None



def _task_loss(cfg: Config, inputs, outputs, labels) -> torch.Tensor:
    if cfg.data.task_type == "cv":
        return F.cross_entropy(outputs, labels)

    if cfg.data.task_type == "llm":
        return outputs.mean()

    if cfg.data.task_type == "speech":
        attention_mask = torch.ones_like(inputs, dtype=torch.long)
        model_name = cfg.data.full_model_name
        input_lengths = get_feat_extract_output_lengths(
            attention_mask.sum(-1), config=AutoConfig.from_pretrained(model_name)
        ).to(torch.long)

        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        log_probs = F.log_softmax(outputs, dim=-1, dtype=torch.float32).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            return F.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                reduction="mean",
            )

    raise NotImplementedError(f"Loss not implemented for task_type={cfg.data.task_type!r}")


@dataclass(frozen=True)
class SweepPoint:
    checkpoint: Path
    # If we can parse a timestamp from the filename; otherwise None.
    timestamp: Optional[datetime]


_CKPT_RE = re.compile(r"^checkpoint_(?P<ts>.+)\.pt$")


def _parse_checkpoint_timestamp(path: Path) -> Optional[datetime]:
    """
    Expected format from our checkpoint saver:
      checkpoint_2026-01-05_04:23:50.355696.pt
    """
    m = _CKPT_RE.match(path.name)
    if not m:
        return None
    ts = m.group("ts")
    # the saver uses datetime.now().isoformat(sep="_")
    try:
        return datetime.fromisoformat(ts.replace("_", "T", 1))
    except ValueError:
        return None


def _iter_checkpoints(checkpoint_dir: Path, *, include_last: bool, recursive: bool) -> list[SweepPoint]:
    if checkpoint_dir.is_file():
        return [SweepPoint(checkpoint=checkpoint_dir, timestamp=_parse_checkpoint_timestamp(checkpoint_dir))]

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Expected a directory or .pt file, got: {checkpoint_dir}")

    pattern = "**/checkpoint_*.pt" if recursive else "checkpoint_*.pt"
    paths = list(checkpoint_dir.glob(pattern))
    points: list[SweepPoint] = []
    for p in paths:
        if p.name == "checkpoint_last.pt" and not include_last:
            continue
        points.append(SweepPoint(checkpoint=p, timestamp=_parse_checkpoint_timestamp(p)))

    def _sort_key(sp: SweepPoint):
        if sp.timestamp is not None:
            return (0, sp.timestamp)
        # Fallback: mtime
        return (1, datetime.fromtimestamp(sp.checkpoint.stat().st_mtime))

    points.sort(key=_sort_key)
    return points


def _evaluate_single_checkpoint(
    *,
    cfg: Config,
    model: torch.nn.Module,
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    dev: torch.device,
) -> Dict[str, Any]:
    _load_checkpoint_into_model(model, checkpoint_path)
    ckpt_stats = _extract_checkpoint_stats(checkpoint_path)

    # Image generation (BigGAN): evaluation is self-contained (no dataset iteration).
    # We intentionally evaluate ONCE per checkpoint, since IS/FID computation is expensive.
    if cfg.data.task_type == "image_gen":
        biggan = model.model_pipeline[0]
        metrics0 = biggan.evaluate(step=0)
        if metrics0 is None:
            enable_eval = getattr(biggan, "enable_eval", None)
            moments = None
            try:
                moments = getattr(biggan, "config", {}).get("eval_moments_file")
            except Exception:
                moments = None
            raise RuntimeError(
                "BigGAN evaluation returned None. This usually means evaluation is disabled "
                f"(biggan.enable_eval={enable_eval}) or the inception moments file is missing/invalid "
                f"(biggan.eval_moments_file={moments!r})."
            )

        # Convert possible numpy scalar types to plain Python floats for JSON/CSV stability.
        fid = float(metrics0.get("FID")) if metrics0.get("FID") is not None else float("nan")
        is_mean = float(metrics0.get("IS_mean")) if metrics0.get("IS_mean") is not None else float("nan")
        is_std = float(metrics0.get("IS_std")) if metrics0.get("IS_std") is not None else float("nan")
        best_is = float(metrics0.get("best_IS")) if metrics0.get("best_IS") is not None else float("nan")
        best_fid = float(metrics0.get("best_FID")) if metrics0.get("best_FID") is not None else float("nan")

        print(
            f"BigGAN metrics: FID={fid}, IS={is_mean}±{is_std}, best_IS={best_is}, best_FID={best_fid}"
        )

        return {
            "split": split,
            # Keep "loss" populated for backward compatibility with existing tooling.
            # For image_gen, interpret "loss" as FID.
            "loss": fid,
            "FID": fid,
            "IS_mean": is_mean,
            "IS_std": is_std,
            "best_IS": best_is,
            "best_FID": best_fid,
            "num_items": 1,
            "num_batches": 1,
            "checkpoint": str(checkpoint_path),
            "device": str(dev),
            "updates": _safe_int(ckpt_stats.get("updates")),
            "examples_processed": _safe_int(ckpt_stats.get("examples_processed")),
        }

    _, val_ds = get_train_val_datasets(cfg.data)


    # On CPU, bf16/fp16 inputs can break conv kernels (and mixed dtypes are unsupported).
    # Force float32 inputs by overriding the collate precision.
    data_cfg = cfg.data
    if dev.type == "cpu":
        try:
            data_cfg = cfg.data.model_copy(update={"precision": "32-true"})
        except Exception:
            # If pydantic version doesn't support model_copy(update=...), fall back to original.
            data_cfg = cfg.data

    cfn = collate_fn(data_cfg, cfg.model_pipeline.pipeline[0])
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=int(num_workers),
        collate_fn=cfn,
    )

    total_loss = 0.0
    total_items = 0
    total_correct = 0

    amp_dtype = None
    if dev.type == "cuda":
        if cfg.data.precision == "fp16-mixed":
            amp_dtype = torch.float16
        elif cfg.data.precision == "bf16-mixed":
            amp_dtype = torch.bfloat16

    with torch.inference_mode():
        for i, (_uids, batch) in enumerate(loader):
            if i >= max_batches:
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
                outputs = model(inputs, labels)
                loss_t = _task_loss(cfg, inputs, outputs, labels)

            bsz = int(labels.shape[0]) if hasattr(labels, "shape") else batch_size
            total_loss += float(loss_t.item()) * bsz
            total_items += bsz

            if cfg.data.task_type == "cv":
                preds = outputs.argmax(dim=-1)
                total_correct += int((preds == labels).sum().item())

    if total_items == 0:
        raise RuntimeError("No batches were evaluated; check dataset access and split name.")

    mean_loss = total_loss / total_items
    metrics: Dict[str, Any] = {
        "split": split,
        "loss": mean_loss,
        "num_items": total_items,
        "num_batches": min(max_batches, math.ceil(total_items / batch_size)),
        "checkpoint": str(checkpoint_path),
        "device": str(dev),
        "updates": _safe_int(ckpt_stats.get("updates")),
        "examples_processed": _safe_int(ckpt_stats.get("examples_processed")),
    }
    if cfg.data.task_type == "cv":
        metrics["accuracy_top1"] = total_correct / total_items
    if cfg.data.task_type == "llm":
        metrics["perplexity"] = float(math.exp(mean_loss)) if mean_loss < 50 else float("inf")
    return metrics


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # union keys across rows for a stable table
    keys: list[str] = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--config-path", type=str, required=True, help="Path to YAML config (e.g. configs/resnet50.yaml)")
@click.option(
    "--sweep-dir",
    type=str,
    required=True,
    help="Directory containing checkpoint_*.pt files (or a single checkpoint .pt file).",
)
@click.option("--split", type=click.Choice(["train", "validation", "test"]), default="validation", show_default=True)
@click.option("--batch-size", type=int, default=None, help="Overrides cfg.diloco.batch_size_per_step for eval.")
@click.option("--max-batches", type=int, default=200, show_default=True, help="Stop after this many batches.")
@click.option(
    "--num-workers",
    type=int,
    default=0,
    show_default=True,
    help="DataLoader workers (0 is simplest and most reproducible across sweeps).",
)
@click.option("--disable-quant", is_flag=True, help="Disable quantization hooks (must match how the checkpoint was trained).")
@click.option("--include-last", is_flag=True, help="Also include checkpoint_last.pt in the sweep (default: skip).")
@click.option("--recursive", is_flag=True, help="Recursively search for checkpoint_*.pt under checkpoint-dir.")
@click.option(
    "--out-dir",
    type=str,
    default=None,
    help="Directory to write metrics + plots. Default: <checkpoint-dir>/eval_sweep_<split>/",
)
@click.option(
    "--x-axis",
    type=click.Choice(["time", "samples", "updates", "index"]),
    default="time",
    show_default=True,
    help="X axis for plots. 'time' is relative (hours since first).",
)
@from_pydantic(Config)
def main(
    config_path: str,
    sweep_dir: str,
    split: str,
    batch_size: Optional[int],
    max_batches: int,
    num_workers: int,
    disable_quant: bool,
    include_last: bool,
    recursive: bool,
    out_dir: Optional[str],
    x_axis: str,
    config: Config,
    **_kwargs,
):
    # Merge YAML base config with CLI overrides provided via pydanclick.
    cfg0 = parse_yaml_file_as(Config, config_path)
    merged = always_merger.merge(cfg0.model_dump(exclude_unset=True), config.model_dump(exclude_unset=True))
    cfg = cfg0.model_validate(merged)

    dev = _resolve_device(cfg.device)

    # For BigGAN checkpoint sweeps we want IS/FID. Training configs often set enable_eval=false
    # (they rely on scripts/evaluate.py instead). Force-enable it here.
    if cfg.data.task_type == "image_gen":
        try:
            cfg.biggan["enable_eval"] = True
            # Ensure the model uses a device that exists on this host.
            cfg.biggan["device"] = str(dev)
        except Exception:
            pass

    ckpt_dir = Path(sweep_dir)
    points = _iter_checkpoints(ckpt_dir, include_last=include_last, recursive=recursive)
    if not points:
        raise FileNotFoundError(f"No checkpoints found under: {ckpt_dir}")

    bs = int(batch_size or cfg.diloco.batch_size_per_step)
    if out_dir is None:
        base = ckpt_dir if ckpt_dir.is_dir() else ckpt_dir.parent
        out = base / f"eval_sweep_{split}"
    else:
        out = Path(out_dir)

    print(f"Config: {config_path}")
    print(f"Split: {split}")
    print(f"Device: {dev}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Num checkpoints: {len(points)}")
    print(f"Batch size: {bs}")
    print(f"Max batches: {max_batches}")
    print(f"Num workers: {num_workers}")
    print(f"Output dir: {out}")

    model = BaselineModel(cfg)
    if not disable_quant:
        model, _avg_only_params = attach_quantizers(model, cfg.quant)
    model.to(dev)
    model.eval()

    results: list[Dict[str, Any]] = []
    for idx, sp in enumerate(points):
        ts = sp.timestamp.isoformat() if sp.timestamp is not None else None
        print(f"[{idx+1}/{len(points)}] Evaluating {sp.checkpoint} (ts={ts})")
        metrics = _evaluate_single_checkpoint(
            cfg=cfg,
            model=model,
            checkpoint_path=sp.checkpoint,
            split=split,
            batch_size=bs,
            max_batches=max_batches,
            num_workers=num_workers,
            dev=dev,
        )
        if sp.timestamp is not None:
            metrics["checkpoint_timestamp"] = sp.timestamp.isoformat()
        results.append(metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))

    _write_jsonl(out / "metrics.jsonl", results)
    _write_csv(out / "metrics.csv", results)
    visualize_eval_sweep.plot_eval_sweep(rows=results, out_dir=out / "plots", x_axis=x_axis, show_values=False)
    print(f"Wrote {len(results)} rows to {out / 'metrics.jsonl'} and plots to {out / 'plots'}")


if __name__ == "__main__":
    main()


