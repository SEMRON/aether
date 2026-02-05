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
from click.core import ParameterSource

from hivemind.utils.logging import get_logger

from orchestrator import BaseOrchestrator
from run_script_utils import run_server_proc, get_public_ip
from distqat.config import Config

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).parent

class Orchestrator(BaseOrchestrator):
    def __init__(
        self,
        config_path: str,
        public_ip=None,
        num_servers=1,
        initial_peers=None,
        expert_indices: Optional[List[int]] = None,
        stage_indices: Optional[List[int]] = None,
    ):
        super().__init__(
            config_path=config_path,
            public_ip=public_ip,
            initial_peers=initial_peers,
            is_monitor=False,  # Workers should not generate new wandb run IDs
        )
        self.num_servers = num_servers
        self.initial_peers = initial_peers
        self.expert_indices = expert_indices
        self.stage_indices = stage_indices

    async def start(self):
        print("ORCHESTRATOR: Starting")

        initial_peers = self.initial_peers

        if not initial_peers or len(initial_peers) == 0:
            raise RuntimeError("No initial peers found, try running again")

        initial_peers_json = json.dumps(initial_peers)
        print("Initial peers JSON:", initial_peers_json)

        # Spawn servers
        for server_idx in range(self.num_servers):
            expert_idx = self.expert_indices[server_idx] if self.expert_indices else None
            if self.stage_indices:
                stage_index = self.stage_indices[server_idx]
            else:
                stage_index = 0 if (len(self.config.model_pipeline.pipeline) == 1 and expert_idx is not None) else None
            # Pass only what's necessary, config file handles the rest
            self.server_procs[f"server_{server_idx}"] = run_server_proc(
                config_path=self.config_path, 
                network_initial_peers=initial_peers_json, 
                public_ip=self.public_ip, 
                idx=expert_idx if expert_idx is not None else server_idx, 
                # This is a workaround to set the expert index to idx if there is only one stage and expert index is provided
                stage_index=stage_index,
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
@click.option("--num-servers", type=int, default=1)
@click.option("--public-ip", type=str, default=None)
@click.option("--config-path", type=str, default="configs/resnet18.yaml")
@click.option("--network-initial-peers", type=str, default=None, help="Comma-separated list of initial peers")
@click.option("--initial-peers-path", type=click.Path(exists=True), default=None, help="Path to file containing comma-separated initial peers")
@click.option("--expert-index", type=int, multiple=True)
@click.option("--stage-index", type=int, multiple=True)
# Exclude network.initial_peers from pydanclick to avoid conflict and validation issues with List[str] vs string input
@from_pydantic(Config, exclude=["network.initial_peers"])
def main(
    num_servers: int,
    public_ip: Optional[str],
    config_path: str,
    network_initial_peers: Optional[str],
    initial_peers_path: Optional[str],
    config: Config,
    expert_index: tuple[int, ...],
    stage_index: tuple[int, ...],
    **kwargs,
):
    cfg = parse_yaml_file_as(Config, config_path)

    base_dict = cfg.model_dump(exclude_unset=True)
    nxt_dict = config.model_dump(exclude_unset=True)

        # Avoid letting Click defaults override YAML values (notably log_dir/checkpoint_dir).
    # The Config model sets a derived log_dir by default; if the user didn't explicitly pass
    # --log-dir/--checkpoint-dir, we must ignore those defaults here.
    ctx = click.get_current_context(silent=True)
    if ctx is not None:
        for field_name in ("log_dir", "checkpoint_dir"):
            try:
                if ctx.get_parameter_source(field_name) == ParameterSource.DEFAULT:
                    nxt_dict.pop(field_name, None)
            except Exception:
                pass
    merged_dict = always_merger.merge(base_dict, nxt_dict)

    merged_cfg = cfg.model_validate(merged_dict)
    
    # Read initial peers from file if path provided
    if initial_peers_path is not None:
        with open(initial_peers_path, "r") as f:
            file_content = f.read().strip()
        merged_cfg.network.initial_peers = [p.strip() for p in file_content.split(",") if p.strip()]
    
    # Command-line initial peers override file-based ones
    if network_initial_peers is not None:
        try:
            parsed_json = json.loads(network_initial_peers)
            if isinstance(parsed_json, list):
                merged_cfg.network.initial_peers = parsed_json
            else:
                merged_cfg.network.initial_peers = [p.strip() for p in network_initial_peers.split(",") if p.strip()]
        except (json.JSONDecodeError, ValueError):
            merged_cfg.network.initial_peers = [p.strip() for p in network_initial_peers.split(",") if p.strip()]
    
    initial_peers = merged_cfg.network.initial_peers
    
    resolved_public_ip = public_ip or get_public_ip()
    
    print("Public IP:", resolved_public_ip)
    print("Num servers:", num_servers)
    print("Initial peers:", initial_peers)

    if expert_index and len(expert_index) != num_servers:
        raise ValueError("expert_index must be provided once per server")

    if stage_index and len(stage_index) != num_servers:
        raise ValueError("stage_index must be provided once per server")

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
                initial_peers=initial_peers,
                expert_indices=list(expert_index) if expert_index else None,
                stage_indices=list(stage_index) if stage_index else None,
            )
            
            # If we successfully retrieved wandb_run_id from DHT, update the temp config file
            # so that spawned server processes can also read it from the config
            if orch.wandb_run_id:
                merged_cfg.wandb_run_id = orch.wandb_run_id
                with open(temp_config_path, "w") as f:
                    yaml.dump(merged_cfg.model_dump(mode='json'), f)
                print(f"Updated temp config with wandb_run_id: {orch.wandb_run_id}")
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
