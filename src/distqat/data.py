#!/bin/env python3

import builtins
import contextlib
import os

from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import datasets, transforms
from ogb.nodeproppred import PygNodePropPredDataset
import torch
from typing import Literal
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, Wav2Vec2Processor

from distqat.utils.biggan.utils import CenterCropLongEdge
from distqat.config import DataConfig, ModelConfig
from distqat.utils.hash import hash64
import numpy as np


@contextlib.contextmanager
def _default_input(default: str = "y"):
    """Temporarily override input() to always return `default`."""
    orig_input = builtins.input
    builtins.input = lambda *args, **kwargs: default
    try:
        yield
    finally:
        builtins.input = orig_input


class CVDataset(IterableDataset):
    def __init__(self, dataset, content_key="image", transform=None):
        self.dataset = dataset
        self.content_key = content_key
        self.transform = transform

    def __iter__(self):
        for ex in self.dataset:
            image = ex[self.content_key]
            image = self.transform(image)
            label = np.asarray(ex["label"], dtype=np.int64)
            image_arr = np.asarray(image, dtype=np.float32)
            uid = hash64(image_arr.tobytes())
            yield uid, {
                "image": image,
                "label": label,
            }


class SequencePackingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len, content_key="Text"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.content_key = content_key
        
    def __iter__(self):
        buffer = []
        for ex in self.dataset:
            # Tokenize the input text if not already tokenized
            text = ex.get(self.content_key, None)
            if text is None:
                continue
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False
            )["input_ids"]

            eos = self.tokenizer.eos_token_id
            if eos is not None and ids and ids[-1] != eos:
                ids.append(eos)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                block = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]

                # ---- UID: hash of token block (int32 -> bytes) ----
                block_arr = np.asarray(block, dtype=np.int32)
                uid = hash64(block_arr.tobytes())

                yield uid, {
                    "input_ids": block,
                    "labels": block.copy(),
                    "attention_mask": [1] * self.seq_len
                }

class SpeechDataset(IterableDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __iter__(self):
        for ex in self.dataset:
            audio = ex["audio"]
            text = ex["text"]
            if "id" in ex:
                uid = hash64(str(ex["id"]).encode())
            else:
                audio_arr = np.asarray(audio["array"], dtype=np.float32)
                uid = hash64(audio_arr.tobytes())
            yield uid, {
                "input_values": self.processor(
                    audio=audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
                ).input_values[0],
                "labels": self.processor(text=text).input_ids,
            }


class GraphDataset(IterableDataset):
    def __init__(self, data, repeat: bool = True, uid_seed: int = 0):
        """
        Yields the full graph in the format expected by the trainer/model pipeline.
        
        Args:
            data: PyTorch Geometric Data object
            repeat: If True, yields the same full-graph sample indefinitely (with different uids)
            uid_seed: Mixed into the UID generation to support multiple independent streams
        """
        self.data = data
        self.repeat = repeat
        self.uid_seed = uid_seed

    def __iter__(self):
        data = self.data
        x = data.x
        edge_index = data.edge_index
        y = data.y

        x_arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
        edge_arr = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        graph_uid = hash64(np.concatenate([x_arr.reshape(-1), edge_arr.reshape(-1)]).tobytes())
        i = 0
        while True:
            uid = hash64(f"{self.uid_seed}:{graph_uid}:{i}".encode())
            yield uid, {"x": x, "edge_index": edge_index, "y": y}
            i += 1
            if not self.repeat:
                break

def cv_collate_fn(batch, precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed"):
    import torch
    dtype = torch.float32
    if precision in ["fp16-mixed", "fp16"]:
        dtype = torch.float16
    elif precision in ["bf16-mixed", "bf16"]:
        dtype = torch.bfloat16
    uids, batch = zip(*batch)
    images = [
        (b["image"].to(dtype=dtype) if torch.is_tensor(b["image"]) else torch.as_tensor(b["image"], dtype=dtype))
        for b in batch
    ]
    
    labels = [torch.as_tensor(b["label"], dtype=torch.long) for b in batch]
    return uids, {
        "inputs": torch.stack(images, dim=0),
        "labels": torch.stack(labels, dim=0),
    }

def llm_collate_fn(batch):
    import torch
    uids, batch = zip(*batch)

    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
    
    return uids, {
        "inputs": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0)
    }

def speech_collate_fn(features, processor):
    uids, features = zip(*features)
    
    # split inputs and labels since they have to be of different lengths and need
    # different padding methods
    input_features = [{"input_values": f["input_values"]} for f in features]
    batch = processor.pad(input_features, padding=True, return_tensors="pt")
    batch["inputs"] = batch.pop("input_values")

    labels_batch = [{"input_ids": f["labels"]} for f in features]
    labels_batch = processor.tokenizer.pad(labels_batch, padding=True, return_tensors="pt")

    # replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    batch["labels"] = labels
    return uids, batch

def graph_collate_fn(batch, hidden_dim: int):
    """
    Collate function for graph data.
    Each item is a full-graph sample dict with keys {"x","edge_index","y"}.
    Returns x and edge_index as a tuple, packed with a leading batch dimension of 1:
      - x: (1, num_nodes, in_dim)
      - edge_index: (1, 2, num_edges)
    """
    import torch
    uids, batch_data = zip(*batch)
    num_nodes = batch_data[0]["x"].shape[0]
    dropout_mask = torch.randint(0, 2, (num_nodes, hidden_dim), dtype=torch.long)

    if len(batch_data) == 1:
        data = batch_data[0]
        x = data["x"].unsqueeze(0)
        edge_index = data["edge_index"].unsqueeze(0)
        labels = data["y"]
        labels = labels.t()
        dropout_mask = dropout_mask.unsqueeze(0)
    else:
        # Full-batch graph training should use DataLoader(batch_size=1). Merging multiple full graphs
        # is both expensive and almost certainly unintended.
        raise ValueError(
            f"graph_collate_fn received batch_size={len(batch_data)} full graphs. "
            "Please set the DataLoader batch_size to 1 for task_type='node_pred'."
        )
    return uids, {
        "inputs": (x, edge_index, dropout_mask),
        "labels": labels,
    }

def collate_fn(data_config: DataConfig, model_config: ModelConfig):
    if data_config.task_type == "cv" or data_config.task_type == "image_gen":
        return lambda batch: cv_collate_fn(batch, data_config.precision)
    elif data_config.task_type == "llm":
        return llm_collate_fn
    elif data_config.task_type == "speech":
        # Create processor once and return a closure
        processor = Wav2Vec2Processor.from_pretrained(data_config.full_model_name)
        return lambda features: speech_collate_fn(features, processor)
    elif data_config.task_type == "node_pred":
        return lambda batch: graph_collate_fn(batch, model_config.hid_dim)
    else:
        return None

def get_train_val_datasets(data_config: DataConfig):
    """
    Get PyTorch DataLoader for the specified dataset.

    Args:
        data_config: DataConfig object containing dataset configuration

    Returns:
        tuple: (train_loader, val_loader) - PyTorch DataLoaders for training and validation
    """
    if data_config.task_type == "cv":
        ds = load_dataset(data_config.dataset_name, split=data_config.dataset_split, streaming=True if data_config.dataset_path is None else False, token=data_config.hf_token, cache_dir=data_config.dataset_path)
        content_key = "image"
        val_split = "validation"
        # For CV tasks, use dataset-specific normalization stats
        if "mnist" in data_config.dataset_name:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ])
            val_split = "test"
        elif "cifar10" in data_config.dataset_name:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 mean and std
            ])
            val_split = "test"
            # different key for cifar10
            content_key = "img"
        elif "imagenet-1k" in data_config.dataset_name:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(232),
                transforms.CenterCrop(data_config.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet mean and std
            ])
        train_dataset = CVDataset(ds, content_key=content_key, transform=transform)

        val_dataset = load_dataset(data_config.dataset_name, split=val_split, streaming=True, token=data_config.hf_token)
        val_dataset = CVDataset(val_dataset, content_key=content_key, transform=transform)

    elif data_config.task_type == "llm":
        if "tiny-shakespeare" in data_config.dataset_name:
            content_key = "Text"
            tokenizer_name = data_config.full_model_name
        elif "pile" in data_config.dataset_name:
            parquet_glob = "hf://datasets/EleutherAI/pile@refs/convert/parquet/all/partial-train/*.parquet"
            dataset=load_dataset(
                    "parquet",
                    data_files={data_config.dataset_split: parquet_glob},
                    split=data_config.dataset_split,
                    streaming=True,
                )
            train_dataset = SequencePackingDataset(
                dataset=dataset,
                tokenizer=AutoTokenizer.from_pretrained(data_config.full_model_name, use_fast=True, model_max_length=int(1e30)),
                seq_len=data_config.seq_len,
                content_key="text",
            )
            return train_dataset, None

        else:
            content_key = "text"
            tokenizer_name = data_config.full_model_name

        ds = load_dataset(data_config.dataset_name, data_config.dataset_config, split=data_config.dataset_split, streaming=True, token=data_config.hf_token)
        train_dataset = SequencePackingDataset(
            dataset=ds,
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=int(1e30)),
            seq_len=data_config.seq_len,
            content_key=content_key,
        )

        val_dataset = None
    elif data_config.task_type == "speech":
        processor = Wav2Vec2Processor.from_pretrained(data_config.full_model_name)
        train_dataset = SpeechDataset(
            load_dataset(data_config.dataset_name, data_config.dataset_config, split=data_config.dataset_split, streaming=True, token=data_config.hf_token),
            processor,
        )
        val_dataset = None
    elif data_config.task_type == "node_pred":
        try:
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data import Data, HeteroData
            from torch_geometric.data.storage import GlobalStorage

            torch.serialization.add_safe_globals([Data, HeteroData, DataEdgeAttr, DataTensorAttr, GlobalStorage])

        except ImportError:
            pass
        # OGB may prompt for dataset download/update via input(); always answer "yes" to avoid hanging.
        with _default_input("y"):
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="data")
        data = dataset[0]

        train_dataset = GraphDataset(data=data, repeat=True, uid_seed=0)
        val_dataset = None

    elif data_config.task_type == "image_gen":
        norm_mean = [0.5,0.5,0.5]
        norm_std = [0.5,0.5,0.5]
        image_size = int(data_config.img_size)
        if 'cifar10' in data_config.dataset_name:
            tfms = []
        else:
            tfms = [
                CenterCropLongEdge(),
            ]
        transform = transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB'))] +
            tfms + 
            [transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        content_key = "img" if 'cifar10' in data_config.dataset_name else "image"
        ds = load_dataset(data_config.dataset_name, split=data_config.dataset_split, streaming=True, token=data_config.hf_token)

        train_dataset = CVDataset(ds, content_key=content_key, transform=transform)
        val_dataset = None
    else:
        raise ValueError(f"Unsupported dataset: {data_config.dataset_name}")

    return train_dataset, val_dataset
