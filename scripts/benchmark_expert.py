#!/usr/bin/env python3
import torch
import time
import click
import sys
from pathlib import Path

# Add src to path to allow importing distqat
sys.path.append(str(Path(__file__).parent.parent / "src"))

from distqat.config import Config, parse_yaml_file_as
from distqat.distributed.server.server import SwarmServer
from hivemind.utils.tensor_descr import BatchTensorDescriptor
from distqat.optimizers import get_diloco_optimizer_cls_kwargs
from hivemind.dht import DHT

def generate_dummy_batch(schema, batch_size, device):
    inputs = []
    for desc in schema:
        # Create a copy of the shape with batch size
        shape = list(desc.shape)
        if shape[0] is None:
            shape[0] = batch_size
        else:
            # If batch dimension is fixed (unlikely for schema), strictly speaking we should respect it
            # But usually it is None. If it's not None, we assume it's the batch dim.
            shape[0] = batch_size
            
        dtype = desc.dtype
        
        if dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            tensor = torch.randn(shape, dtype=dtype, device=device)
        elif dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
            # For integer types, assume they might be indices or similar. 
            # A safe bet for indices is usually small numbers.
            tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
        elif dtype == torch.bool:
            tensor = torch.randint(0, 2, shape, dtype=dtype, device=device)
        else:
             # Fallback
             tensor = torch.zeros(shape, dtype=dtype, device=device)
        inputs.append(tensor)
    return tuple(inputs)

@click.command()
@click.option("--config-path", type=str, required=True, help="Path to config yaml")
@click.option("--batch-size", type=int, default=32, help="Batch size for benchmarking")
@click.option("--num-batches", type=int, default=100, help="Number of batches to run")
@click.option("--warmup-batches", type=int, default=10, help="Number of warmup batches")
@click.option("--device", type=str, default=None, help="Device to run on (e.g. cuda, cpu)")
@click.option("--stage-index", type=int, default=0, help="Index of stage to benchmark")
def main(config_path, batch_size, num_batches, warmup_batches, device, stage_index):
    try:
        cfg = parse_yaml_file_as(Config, config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Override device if specified
    if device:
        cfg.device = device
    
    # Ensure device is valid
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU")
        cfg.device = "cpu"
        
    print(f"Loading server from {config_path}...")

    dht = DHT(
        start=True,
        initial_peers=cfg.network.initial_peers,
        host_maddrs=cfg.network.host_maddrs,
        announce_maddrs=cfg.network.announce_maddrs,
    )
    
    models, stages = [], []
    for pipeline_step_cfg in cfg.model_pipeline.pipeline:
        model, stage = pipeline_step_cfg.model_name.split(".")
        models.append(model)
        stages.append(stage)
    expert_cls = f"{models[stage_index]}.{stages[stage_index]}"
    expert_uid = f"{stages[stage_index]}.0.0.0"

    optim_cls, optim_kwargs = get_diloco_optimizer_cls_kwargs(f"{cfg.experiment_prefix}_{stage_index}", cfg.diloco)
    
    # Create server
    try:
        server = SwarmServer.create(
            start=False,
            initial_peers=cfg.network.initial_peers,
            host_maddrs=cfg.network.host_maddrs,
            device=cfg.device,
            expert_cls=expert_cls,
            expert_uids=[expert_uid],
            hidden_dim=cfg.model_pipeline.pipeline[stage_index].hid_dim,
            optim_cls=optim_cls,
            optim_kwargs=optim_kwargs,
            min_batch_size=1,
            max_batch_size=64,
            quant_config=None,
            dht=dht,
            cfg=cfg,
            stage_index=stage_index,
        )
    except Exception as e:
        print(f"Failed to create server: {e}")
        return
    
    if not server.experts:
        print("No experts found in server.")
        return

    expert_items = list(server.experts.items())
    # if expert_index >= len(expert_items):
    #     print(f"Expert index {expert_index} out of range. Only {len(expert_items)} experts available.")
    #     return

    expert_name, expert_backend = expert_items[0]
    print(f"\nBenchmarking expert: {expert_name}")
    print(f"Class: {expert_backend.expert.__class__.__name__}")
    print(f"Device: {cfg.device}")
    print(f"Batch Size: {batch_size}")
    
    device_obj = torch.device(cfg.device)
    
    # Prepare dummy inputs
    inputs = generate_dummy_batch(expert_backend.args_schema, batch_size, device_obj)
    
    print(f"\nWarming up for {warmup_batches} batches...")
    try:
        with torch.no_grad():
            for _ in range(warmup_batches):
                expert_backend.forward(*inputs)
        
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
            
        print(f"Running benchmark for {num_batches} batches...")
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_batches):
                expert_backend.forward(*inputs)
                
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
                
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_batches
        throughput = batch_size / avg_time
        
        print("-" * 40)
        print(f"Total time:       {total_time:.4f} s")
        print(f"Avg time/batch:   {avg_time*1000:.2f} ms")
        print(f"Throughput:       {throughput:.2f} samples/s")
        print("-" * 40)
        
    except RuntimeError as e:
        print(f"RuntimeError during benchmark: {e}")
        if "out of memory" in str(e):
             print("Try reducing the batch size.")

if __name__ == "__main__":
    main()

