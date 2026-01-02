import os

# Set cache directory BEFORE importing datasets/transformers
os.environ.setdefault("HF_DATASETS_CACHE", "/simdata/pahrendt/datasets")

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from distqat.config import DataConfig, ModelConfig
from distqat.data import collate_fn, get_train_val_datasets
from distqat.models import GPTNeoFull


def main():
    parser = argparse.ArgumentParser(description="Quick GPT-Neo throughput/sanity training loop (like test_resnet_50.py)")
    parser.add_argument("--full-model-name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--dataset-name", type=str, default="EleutherAI/pile")
    parser.add_argument("--dataset-config", type=str, default="default")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    # Number of *optimizer* steps (not micro-batches)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grad-accumulation-steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_cfg = DataConfig(
        task_type="llm",
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        full_model_name=args.full_model_name,
    )

    model = GPTNeoFull(full_model_name=data_cfg.full_model_name).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    ds, _ = get_train_val_datasets(data_cfg)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,  # IterableDataset
        collate_fn=collate_fn(data_cfg, ModelConfig()),
        persistent_workers=data_cfg.num_workers > 0,
        prefetch_factor=2 if data_cfg.num_workers > 0 else None,
    )

    model.train()
    start_time = time.time()
    grad_accumulation_steps = max(1, int(args.grad_accumulation_steps))
    micro_step = 0
    opt_step = 0
    optimizer.zero_grad(set_to_none=True)
    with tqdm(loader, desc="GPT-Neo", unit="batch") as pbar:
        for i, (uids, batch) in enumerate(pbar):
            _ = uids  # unused
            micro_step += 1
            input_ids = batch["inputs"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)

            logits = model(input_ids)  # [B, T, vocab]

            # Next-token prediction (matches Trainer.task_type_loss for "llm")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.float().permute(0, 2, 1),
                shift_labels,
                reduction="mean",
            )

            (loss / grad_accumulation_steps).backward()

            if micro_step % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                # if opt_step % 10 == 0:
                pbar.set_postfix({"loss": float(loss.detach().cpu()), "opt_step": opt_step})
                if opt_step >= args.steps:
                    break

    end_time = time.time()
    print(f"Optimizer steps: {opt_step} | Micro-batches: {micro_step} | Total time: {end_time - start_time:.2f}s")
    print(f"Avg time/optimizer step: {(end_time - start_time) / max(1, opt_step):.4f}s")


if __name__ == "__main__":
    main()



