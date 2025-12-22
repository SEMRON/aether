#!/usr/bin/env python3

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import click
import torch
import torch.nn.functional as F
from deepmerge import always_merger
from pydanclick import from_pydantic
from pydantic_yaml import parse_yaml_file_as
from torch.utils.data import DataLoader
from transformers import AutoConfig

from distqat.attach import attach_quantizers
from distqat.config import Config
from distqat.data import (
    CVDataset,
    SequencePackingDataset,
    SpeechDataset,
    collate_fn,
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
        print(f"WARNING: Missing keys when loading checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(
            f"WARNING: Unexpected keys when loading checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )


def _default_checkpoint_from_cfg(cfg: Config) -> Path:
    if cfg.checkpoint_dir is None:
        raise ValueError("Config.checkpoint_dir is None; pass --checkpoint-path explicitly.")
    return Path(cfg.checkpoint_dir) / "baseline" / "checkpoint_last.pt"


def _find_checkpoint_path(cfg: Config, checkpoint_path: Optional[str]) -> Path:
    if checkpoint_path is None:
        ckpt = _default_checkpoint_from_cfg(cfg)
    else:
        ckpt = Path(checkpoint_path)
        if ckpt.is_dir():
            # Common layouts:
            # - checkpoints/<exp>/baseline/checkpoint_last.pt
            # - checkpoints/<exp>/checkpoint_last.pt
            candidate_a = ckpt / "baseline" / "checkpoint_last.pt"
            candidate_b = ckpt / "checkpoint_last.pt"
            ckpt = candidate_a if candidate_a.exists() else candidate_b
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _build_eval_dataset(cfg: Config, split: str):
    """
    Build a split-specific dataset consistent with training preprocessing.

    Notes:
    - For CIFAR-10 on HF, the 'validation' split is called 'test'. We map validation->test.
    """
    from datasets import load_dataset
    from torchvision import transforms
    from transformers import AutoTokenizer, Wav2Vec2Processor

    data_cfg = cfg.data.model_copy()
    model_cfg = cfg.model_pipeline.pipeline[0]

    if data_cfg.task_type == "cv":
        ds_name = data_cfg.dataset_name
        use_split = split
        if "cifar10" in ds_name and split == "validation":
            use_split = "test"

        content_key = "image"
        if "mnist" in ds_name:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        elif "cifar10" in ds_name:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            content_key = "img"
        elif "imagenet-1k" in ds_name:
            transform = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(232),
                    transforms.CenterCrop(data_cfg.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            raise ValueError(f"Unsupported cv dataset_name: {ds_name}")

        ds = load_dataset(ds_name, split=use_split, streaming=True, token=data_cfg.hf_token)
        return CVDataset(ds, content_key=content_key, transform=transform)

    if data_cfg.task_type == "llm":
        ds = load_dataset(
            data_cfg.dataset_name,
            data_cfg.dataset_config,
            split=split,
            streaming=True,
            token=data_cfg.hf_token,
        )
        tokenizer_name = data_cfg.full_model_name
        content_key = "text"
        if "tiny-shakespeare" in data_cfg.dataset_name:
            content_key = "Text"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=int(1e30))
        return SequencePackingDataset(ds, tokenizer=tokenizer, seq_len=data_cfg.seq_len, content_key=content_key)

    if data_cfg.task_type == "speech":
        processor = Wav2Vec2Processor.from_pretrained(data_cfg.full_model_name)
        ds = load_dataset(
            data_cfg.dataset_name,
            data_cfg.dataset_config,
            split=split,
            streaming=True,
            token=data_cfg.hf_token,
        )
        return SpeechDataset(ds, processor=processor)

    if data_cfg.task_type in ("node_pred", "image_gen", "rl"):
        raise NotImplementedError(f"Offline eval is not implemented for task_type={data_cfg.task_type!r} yet.")

    raise ValueError(f"Unknown task_type: {data_cfg.task_type}")


def _task_loss(cfg: Config, inputs, outputs, labels) -> torch.Tensor:
    if cfg.data.task_type == "cv":
        return F.cross_entropy(outputs, labels)

    if cfg.data.task_type == "llm":
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none").mean()
        return loss

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


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--config-path", type=str, required=True, help="Path to YAML config (e.g. configs/resnet18.yaml)")
@click.option(
    "--checkpoint-path",
    type=str,
    default=None,
    help="Checkpoint file (.pt) or directory. If omitted, uses <checkpoint_dir>/baseline/checkpoint_last.pt from config.",
)
@click.option("--split", type=click.Choice(["train", "validation", "test"]), default="validation", show_default=True)
@click.option("--batch-size", type=int, default=None, help="Overrides cfg.diloco.batch_size_per_step for eval.")
@click.option("--max-batches", type=int, default=200, show_default=True, help="Stop after this many batches.")
@click.option("--disable-quant", is_flag=True, help="Disable quantization hooks (must match how the checkpoint was trained).")
@click.option("--output-json", type=str, default=None, help="Optional path to write metrics as JSON.")
@from_pydantic(Config)
def main(
    config_path: str,
    checkpoint_path: Optional[str],
    split: str,
    batch_size: Optional[int],
    max_batches: int,
    disable_quant: bool,
    output_json: Optional[str],
    config: Config,
    **_kwargs,
):
    # Merge YAML base config with CLI overrides provided via pydanclick.
    cfg0 = parse_yaml_file_as(Config, config_path)
    merged = always_merger.merge(cfg0.model_dump(exclude_unset=True), config.model_dump(exclude_unset=True))
    cfg = cfg0.model_validate(merged)

    dev = _resolve_device(cfg.device)
    ckpt = _find_checkpoint_path(cfg, checkpoint_path)
    bs = int(batch_size or cfg.diloco.batch_size_per_step)

    print(f"Config: {config_path}")
    print(f"Split: {split}")
    print(f"Device: {dev}")
    print(f"Checkpoint: {ckpt}")
    print(f"Batch size: {bs}")

    model = BaselineModel(cfg)
    if not disable_quant:
        model, _avg_only_params = attach_quantizers(model, cfg.quant)
    model.to(dev)
    model.eval()

    _load_checkpoint_into_model(model, ckpt)

    ds = _build_eval_dataset(cfg, split=split)
    cfn = collate_fn(cfg.data, cfg.model_pipeline.pipeline[0])
    loader = DataLoader(
        ds,
        batch_size=bs,
        num_workers=int(cfg.data.num_workers),
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
                outputs = model(inputs)
                loss_t = _task_loss(cfg, inputs, outputs, labels)

            bsz = int(labels.shape[0]) if hasattr(labels, "shape") else bs
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
        "num_batches": min(max_batches, math.ceil(total_items / bs)),
        "checkpoint": str(ckpt),
        "device": str(dev),
    }
    if cfg.data.task_type == "cv":
        metrics["accuracy_top1"] = total_correct / total_items
    if cfg.data.task_type == "llm":
        metrics["perplexity"] = float(math.exp(mean_loss)) if mean_loss < 50 else float("inf")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()


