### Convenience run script to start a number of servers based on a config file

# Usage:
# python start_servers.py [--config-path configs/<config_file>.yaml] [--public-ip <public_ip>] [--num-servers <num_servers>] [overrides...]
# Example: python start_servers.py --num-servers 2 --network-initial-peers /ip4/1.2.3.4/tcp/1234 --diloco-batch-size-per-step 32

import asyncio
import subprocess
import os
import signal
import time
import json
import click
import tempfile
import yaml
from typing import Optional, List
from pathlib import Path
from deepmerge import always_merger
from pydanclick import from_pydantic
from pydantic_yaml import parse_yaml_file_as

from hivemind.utils.logging import get_logger

from orchestrator import BaseOrchestrator
from run_script_utils import run_server_proc, get_public_ip
from distqat.config import Config

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).parent
DISABLE_QUANT = True

class Orchestrator(BaseOrchestrator):
    def __init__(self, config_path: str, public_ip=None, num_servers=1, initial_peers=None):
        super().__init__(
            config_path=config_path,
            public_ip=public_ip,
            disable_quant=DISABLE_QUANT,
        )
        self.num_servers = num_servers
        self.initial_peers = initial_peers

    async def start(self):
        print("ORCHESTRATOR: Starting")

        initial_peers = self.initial_peers

        if not initial_peers or len(initial_peers) == 0:
            raise RuntimeError("No initial peers found, try running again")

        initial_peers_json = json.dumps(initial_peers)
        print("Initial peers JSON:", initial_peers_json)

        # Spawn servers
        for idx in range(self.num_servers):
            # Pass only what's necessary, config file handles the rest
            self.server_procs[f"server_{idx}"] = run_server_proc(
                config_path=self.config_path, 
                network_initial_peers=initial_peers_json, 
                public_ip=self.public_ip, 
                idx=idx, 
                disable_quant=self.disable_quant,
                wandb_run_id=self.wandb_run_id
            )
            
            # Give servers time to write their expert UID to the DHT
            time.sleep(3)

    async def wait(self):
        print("ORCHESTRATOR: Waiting for children to finish")

        # Wait for all server processes to finish
        while True:
            if all(p.poll() is not None for p in self.server_procs.values()):
                break

@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--num-servers", type=int, default=2)
@click.option("--public-ip", type=str, default=None)
@click.option("--config-path", type=str, default="configs/resnet18.yaml")
@click.option("--network-initial-peers", type=str, default=None, help="Comma-separated list of initial peers")
# Exclude network.initial_peers from pydanclick to avoid conflict and validation issues with List[str] vs string input
@from_pydantic(Config, exclude=["network.initial_peers"])
def main(num_servers: int, public_ip: Optional[str], config_path: str, network_initial_peers: Optional[str], config: Config, **kwargs):
    cfg = parse_yaml_file_as(Config, config_path)

    base_dict = cfg.model_dump(exclude_unset=True)
    nxt_dict = config.model_dump(exclude_unset=True)
    merged_dict = always_merger.merge(base_dict, nxt_dict)

    merged_cfg = cfg.model_validate(merged_dict)
    
    # Handle legacy comma-separated string for initial peers
    if network_initial_peers is not None:
        merged_cfg.network.initial_peers = [p.strip() for p in network_initial_peers.split(",") if p.strip()]
    
    initial_peers = merged_cfg.network.initial_peers
    
    resolved_public_ip = public_ip or get_public_ip()
    
    print("Public IP:", resolved_public_ip)
    print("Num servers:", num_servers)
    print("Initial peers:", initial_peers)

    # Dump merged config to temp file
    fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="distqat_config_")
    os.close(fd)
    
    try:
        with open(temp_config_path, "w") as f:
            yaml.dump(merged_cfg.model_dump(mode='json'), f)
        
        async def run():
            orch = Orchestrator(
                config_path=temp_config_path, 
                public_ip=resolved_public_ip, 
                num_servers=num_servers, 
                initial_peers=initial_peers
            )
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
