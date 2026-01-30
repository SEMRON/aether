#!/bin/env python3

import builtins
import contextlib
import os

from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import datasets, transforms
from ogb.nodeproppred import PygNodePropPredDataset
import torch
from typing import Literal, Optional, List
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, Wav2Vec2Processor

from distqat.utils.biggan.utils import CenterCropLongEdge
from distqat.config import DataConfig, ModelConfig
from distqat.utils.hash import hash64
from distqat.config import Config

import numpy as np
from distqat.utils.logging import get_logger

logger = get_logger(__name__)

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
            if torch.is_tensor(image) and image.is_contiguous():
                uid = hash64(image.numpy().tobytes())
            else:
                image_arr = np.asarray(image, dtype=np.float32)
                uid = hash64(image_arr.tobytes())
                del image_arr
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


class GraphSamplerDataset(IterableDataset):
    """
    Yields mini-batches of sampled subgraphs using NeighborLoader from PyTorch Geometric.

    This dataset uses neighbor sampling for efficient training on large graphs. Each batch
    contains a subgraph sampled around the target nodes, with neighbors sampled at each
    hop according to num_neighbors.
    """

    def __init__(
        self,
        data,
        input_nodes: torch.Tensor,
        num_neighbors: List[int],
        batch_size: int = 512,
        shuffle: bool = True,
        repeat: bool = True,
        uid_seed: int = 0,
    ):
        """
        Args:
            data: PyTorch Geometric Data object with x, edge_index, y attributes
            input_nodes: Tensor of node indices to sample from (e.g., train/val/test split)
            num_neighbors: List of number of neighbors to sample at each hop, e.g. [10, 10]
            batch_size: Number of target nodes per mini-batch
            shuffle: Whether to shuffle the input nodes before each epoch
            repeat: If True, loops indefinitely over the dataset
            uid_seed: Seed mixed into UID generation for reproducibility
        """
        from torch_geometric.loader import NeighborLoader

        self.data = data
        self.input_nodes = input_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.uid_seed = uid_seed

        # Create the NeighborLoader
        self._loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def __iter__(self):
        i = 0
        while True:
            for batch in self._loader:
                # batch is a Data object with sampled subgraph
                # batch.n_id contains the original node indices in the sampled subgraph
                # batch.batch_size gives the number of target (seed) nodes
                x = batch.x
                edge_index = batch.edge_index
                y = batch.y
                n_id = batch.n_id
                target_size = batch.batch_size  # number of seed nodes

                # Generate a unique ID for this batch
                batch_bytes = f"{self.uid_seed}:{i}:{n_id.tolist()}".encode()
                uid = hash64(batch_bytes)

                yield uid, {
                    "x": x,
                    "edge_index": edge_index,
                    "y": y,
                    "n_id": n_id,
                    "target_size": target_size,
                }
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
    Collate function for sampled subgraph data from NeighborLoader.

    Each item is a mini-batch sample dict with keys {"x", "edge_index", "y", "n_id", "target_size"}.
    Returns the subgraph with a leading batch dimension of 1:
      - x: (1, num_sampled_nodes, in_dim)
      - edge_index: (1, 2, num_sampled_edges)
      - dropout_mask: (1, num_sampled_nodes, hidden_dim)
      - labels: (1, target_size) - only labels for target nodes

    The first `target_size` nodes in x are the target (seed) nodes whose labels we want to predict.
    """
    import torch
    uids, batch_data = zip(*batch)

    if len(batch_data) != 1:
        raise ValueError(
            f"graph_collate_fn received batch_size={len(batch_data)} samples. "
            "Please set the DataLoader batch_size to 1 for task_type='node_pred'."
        )

    data = batch_data[0]
    num_sampled_nodes = data["x"].shape[0]
    target_size = data["target_size"]

    x = data["x"].unsqueeze(0)
    edge_index = data["edge_index"].unsqueeze(0)
    # Labels for target nodes only (first target_size nodes in the sampled subgraph)
    labels = data["y"][:target_size].squeeze(-1).unsqueeze(0)
    dropout_mask = torch.randint(0, 2, (1, num_sampled_nodes, hidden_dim), dtype=torch.int8)

    return uids, {
        "inputs": (x, edge_index, dropout_mask),
        "labels": labels,
        "target_size": target_size,
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

        if data_config.dataset_iid_path is not None:
            # Stream Parquet files directly from S3
            # Expects path like "s3://bucket/imagenet-1k-iid" containing *.parquet files
            parquet_pattern = f"{data_config.dataset_iid_path}/*.parquet"
            logger.info(f"Streaming from S3: {parquet_pattern}")
            ds = load_dataset(
                "parquet",
                data_files={"train": parquet_pattern},
                split="train",
                streaming=True,
                storage_options={"anon": False},  # Use AWS credentials
            )
        else:
            ds = load_dataset(data_config.dataset_name, split=data_config.dataset_split, streaming=True if data_config.dataset_path is None else False, token=data_config.hf_token, cache_dir=data_config.dataset_path)
        content_key = "image"
        val_split = data_config.dataset_split_validation
        # For CV tasks, use dataset-specific normalization stats
        if "mnist" in data_config.dataset_name:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ])
            val_split = data_config.dataset_split_validation
        elif "cifar10" in data_config.dataset_name:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 mean and std
            ])
            val_split = data_config.dataset_split_validation
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

        if data_config.dataset_path is not None:
            ds = ds.to_iterable_dataset(num_shards=data_config.dataset_num_shards)
        ds = ds.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
        train_dataset = CVDataset(ds, content_key=content_key, transform=transform)

        val_dataset = load_dataset(data_config.dataset_name, split=val_split, streaming=True, token=data_config.hf_token)
        val_dataset = val_dataset.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
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
            dataset = dataset.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
            train_dataset = SequencePackingDataset(
                dataset=dataset,
                tokenizer=AutoTokenizer.from_pretrained(data_config.full_model_name, use_fast=True, model_max_length=int(1e30)),
                seq_len=data_config.seq_len,
                content_key="text",
            )
            val_dataset = load_dataset(data_config.dataset_name, data_config.dataset_config, split=data_config.dataset_split_validation, streaming=True, token=data_config.hf_token)
            val_dataset = val_dataset.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
            val_dataset = SequencePackingDataset(
                dataset=val_dataset,
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=int(1e30)),
                seq_len=data_config.seq_len,
                content_key=content_key,
            )

        else:
            content_key = "text"
            tokenizer_name = data_config.full_model_name


        ds = load_dataset(data_config.dataset_name, data_config.dataset_config, split=data_config.dataset_split, streaming=True, token=data_config.hf_token)
        ds = ds.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
        train_dataset = SequencePackingDataset(
            dataset=ds,
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=int(1e30)),
            seq_len=data_config.seq_len,
            content_key=content_key,
        )
        val_dataset = load_dataset(data_config.dataset_name, data_config.dataset_config, split=data_config.dataset_split_validation, streaming=True, token=data_config.hf_token)
        val_dataset = val_dataset.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
        val_dataset = SequencePackingDataset(
            dataset=val_dataset,
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=int(1e30)),
            seq_len=data_config.seq_len,
            content_key=content_key,
        )

    elif data_config.task_type == "speech":
        processor = Wav2Vec2Processor.from_pretrained(data_config.full_model_name)
        dataset = load_dataset(data_config.dataset_name, data_config.dataset_config, split=data_config.dataset_split, streaming=True, token=data_config.hf_token)
        dataset = dataset.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
        train_dataset = SpeechDataset(
            dataset=dataset,
            processor=processor,
        )
        val_dataset = load_dataset(data_config.dataset_name, data_config.dataset_config, split=data_config.dataset_split_validation, streaming=True, token=data_config.hf_token)
        val_dataset = val_dataset.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
        val_dataset = SpeechDataset(
            dataset=val_dataset,
            processor=processor,
        )
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
            dataset = PygNodePropPredDataset(name=data_config.dataset_name, root="data")
        data = dataset[0]

        split_idx = dataset.get_idx_split()
        train_idx = split_idx[data_config.dataset_split]
        val_idx = split_idx[data_config.dataset_split_validation]

        train_dataset = GraphSamplerDataset(
            data=data,
            input_nodes=train_idx,
            num_neighbors=data_config.neighbor_sample_sizes,
            batch_size=data_config.neighbor_batch_size,
            shuffle=True,
            repeat=True,
            uid_seed=0,
        )
        val_dataset = GraphSamplerDataset(
            data=data,
            input_nodes=val_idx,
            num_neighbors=data_config.neighbor_sample_sizes,
            batch_size=data_config.neighbor_batch_size,
            shuffle=False,
            repeat=False,
            uid_seed=1,
        )

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
        ds = ds.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
        train_dataset = CVDataset(ds, content_key=content_key, transform=transform)
        val_dataset = load_dataset(data_config.dataset_name, data_config.dataset_config, split=data_config.dataset_split_validation, streaming=True, token=data_config.hf_token)
        val_dataset = val_dataset.shuffle(buffer_size=data_config.shuffle_buffer_size, seed=data_config.shuffle_seed)
        val_dataset = CVDataset(val_dataset, content_key=content_key, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {data_config.dataset_name}")

    return train_dataset, val_dataset


def get_dataloader(config: Config, dataset: IterableDataset):
        # Get the collate function and verify it's not None
        cfn = collate_fn(config.data, config.model_pipeline.pipeline[0])
        if cfn is None:
            logger.warning(f"collate_fn returned None for task_type={config.data.task_type}. Using default collation which may cause issues.")
        else:
            logger.debug(f"Using custom collate_fn for task_type={config.data.task_type}")
        
        # Note: pin_memory=True converts tuples to lists, which breaks node_pred and other
        # task types that pass tuple inputs. Disable pin_memory for these task types.
        use_pin_memory = config.data.task_type not in ("node_pred",)
        
        loader = DataLoader(
            dataset,                                 # yields (uid, sample) or sample
            batch_size=config.diloco.batch_size_per_step,
            num_workers=config.data.num_workers,
            pin_memory=use_pin_memory,
            drop_last=True,
            shuffle=False,
            collate_fn=cfn,
            # persistent_workers=config.data.num_workers > 0,
            prefetch_factor=2 if config.data.num_workers > 0 else None,
        )
        return iter(loader)