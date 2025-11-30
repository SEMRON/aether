### Convenience run script to start a monitor and a client based on a config file
# Usage:
# python start_trainer_client.py [--config-path configs/<config_file>.yaml] [--public-ip <public_ip>] [overrides...]

# Public IP is the public IP address of the machine running the script. If not provided, it will be automatically detected.
# Config path is the path to the config file. If not provided, it will default to `configs/resnet18.yaml`.

import asyncio
import subprocess
import time
import json
import click
import tempfile
import yaml
import os
from pathlib import Path
from deepmerge import always_merger
from pydanclick import from_pydantic
from pydantic_yaml import parse_yaml_file_as
from typing import Optional

from hivemind.utils.logging import get_logger

from orchestrator import BaseOrchestrator
from run_script_utils import run_monitor_proc, run_client_proc, create_initial_peers_file, wait_for_initial_peers, get_public_ip, ensure_no_leftover_distqat_processes
from distqat.config import Config

logger = get_logger(__name__)

DISABLE_QUANT = True

class Orchestrator(BaseOrchestrator):
    def __init__(self, config_path: str, public_ip=None):
        super().__init__(
            config_path=config_path,
            public_ip=public_ip,
            disable_quant=DISABLE_QUANT,
        )


    async def start(self):
        ensure_no_leftover_distqat_processes()
        print("ORCHESTRATOR: Starting")

        initial_peers_path = create_initial_peers_file(log_dir=self.config.log_dir)

        self.monitor_proc = run_monitor_proc(config_path=self.config_path, refresh_period=2, store_ip_addresses_path=str(initial_peers_path), public_ip=self.public_ip, wandb_run_id=self.wandb_run_id)

        initial_peers = wait_for_initial_peers(initial_peers_path)

        # Spawn client, which in turn spawns trainers dynamically based on complete pipelines
        self.client_proc = run_client_proc(config_path=self.config_path, refresh_period=5, network_initial_peers=initial_peers, public_ip=self.public_ip, disable_quant=self.disable_quant, wandb_run_id=self.wandb_run_id)


    async def wait(self):
        print("ORCHESTRATOR: Waiting for children to finish")

        # Wait for the client process, which manages the distributed trainers
        # Finished training when all trainers are completed
        while True:
            if self.client_proc.poll() is not None:
                print(f"ORCHESTRATOR: Client exited with code {self.client_proc.returncode}")
                break
            
            done = {n: p for n, p in self.trainer_procs.items() if p.poll() is not None}
            for label, p in done.items():
                print(f"ORCHESTRATOR: Trainer {label} exited with code {p.returncode}")
            
            try:
                self.client_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                pass

@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--public-ip", type=str, default=None)
@click.option("--config-path", type=str, default="configs/resnet18.yaml")
@click.option("--network-initial-peers", type=str, default=None, help="Comma-separated list of initial peers")
@from_pydantic(Config, exclude=["network.initial_peers"])
def main(public_ip: Optional[str], config_path: str, network_initial_peers: Optional[str], config: Config, **kwargs):
    cfg = parse_yaml_file_as(Config, config_path)

    base_dict = cfg.model_dump(exclude_unset=True)
    nxt_dict = config.model_dump(exclude_unset=True)
    merged_dict = always_merger.merge(base_dict, nxt_dict)
    merged_cfg = cfg.model_validate(merged_dict)
    
    # Handle legacy comma-separated string for initial peers
    if network_initial_peers is not None:
        merged_cfg.network.initial_peers = [p.strip() for p in network_initial_peers.split(",") if p.strip()]
        
    resolved_public_ip = public_ip or get_public_ip()
    
    print("Public IP:", resolved_public_ip)

    # Dump merged config to temp file
    fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="distqat_config_")
    os.close(fd)
    
    try:
        with open(temp_config_path, "w") as f:
            yaml.dump(merged_cfg.model_dump(mode='json'), f)
        
        async def run():
            orch = Orchestrator(config_path=temp_config_path, public_ip=resolved_public_ip)
            try:
                await orch.start()
                await orch.wait()
                print("Training finished. Shutting down...")
            except KeyboardInterrupt:
                print("Keyboard interrupt. Shutting down...")
            finally:
                await orch.shutdown()

        asyncio.run(run())
        
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    print("Done")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
