import asyncio
import subprocess
import os
import tempfile
import yaml
from typing import Optional

import click
from deepmerge import always_merger
from pydanclick import from_pydantic
from pydantic_yaml import parse_yaml_file_as

from hivemind.utils.logging import get_logger

from orchestrator import BaseOrchestrator
from run_script_utils import (
    run_monitor_proc,
    run_client_proc,
    wait_for_initial_peers,
    create_initial_peers_file,
    clear_data_server_log,
    run_baseline_model_trainer_proc,
    run_server_proc,
    wait_for_data_server_ready,
    get_public_ip,
    ensure_no_leftover_distqat_processes,
    is_wandb_logged_in,
)
from distqat.config import Config

if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ and os.environ["HF_HUB_ENABLE_HF_TRANSFER"] != "0":
    raise RuntimeError("HF_HUB_ENABLE_HF_TRANSFER is set to 1, please set it to 0 in your environment variables")




logger = get_logger(__name__)

DISABLE_QUANT = False

class Orchestrator(BaseOrchestrator):
    def __init__(self, config_path: str, public_ip=None, num_servers=1):
        super().__init__(
            config_path=config_path,
            public_ip=public_ip,
            disable_quant=DISABLE_QUANT,
        )
        self.num_servers = num_servers
    async def start(self):
        ensure_no_leftover_distqat_processes()

        if not is_wandb_logged_in() and self.config.wandb_project:
            raise RuntimeError("Wandb is not logged in, please login to wandb using the wandb login command or set the wandb_project to None through the config file or command line argument")
        print("ORCHESTRATOR: Starting")

        initial_peers_path = create_initial_peers_file(log_dir=self.config.log_dir)

        # Start monitor
        self.monitor_proc = run_monitor_proc(config_path=self.config_path, refresh_period=2, store_ip_addresses_path=str(initial_peers_path), public_ip=self.public_ip, wandb_run_id=self.wandb_run_id)


        initial_peers_json = wait_for_initial_peers(initial_peers_path)


        # Ensure stale data_server.log does not cause a false-positive readiness
        ds_log_path = clear_data_server_log(log_dir=self.config.log_dir)
        
        # Spawn client, which in turn spawns trainers dynamically based on complete pipelines
        self.client_proc = run_client_proc(config_path=self.config_path, refresh_period=5, network_initial_peers=initial_peers_json, public_ip=self.public_ip, disable_quant=self.disable_quant)

        # Wait for data server to be ready (the client starts it)
        # We consider it ready when the log contains the startup line from start_manager(...)
        wait_for_data_server_ready(client_proc=self.client_proc, ds_log_path=ds_log_path, deadline=20)


        # Spawn baseline model trainer
        print(f"ORCHESTRATOR: Spawning baseline model trainer")
        self.baseline_model_trainer_proc = run_baseline_model_trainer_proc(config_path=self.config_path, network_initial_peers=initial_peers_json, public_ip=self.public_ip, disable_quant=self.disable_quant, log_dir=self.config.log_dir)

        print(f"ORCHESTRATOR: Starting to spawn servers")

        # Spawn servers
        for expert_idx in range(self.num_servers):                
            for stage_idx,  pipeline_step_cfg in enumerate(self.config.model_pipeline.pipeline):
                _, stage = pipeline_step_cfg.model_name.split(".")
                proc = run_server_proc(config_path=self.config_path, network_initial_peers=initial_peers_json, public_ip=self.public_ip, idx=expert_idx, stage_index=stage_idx, disable_quant=self.disable_quant)
                self.server_procs[f"server_{stage}_{expert_idx}"] = proc

        print(f"ORCHESTRATOR: Servers spawned")

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
@click.option("--num-servers", type=int, default=1)
@from_pydantic(Config)
def main(public_ip: Optional[str], config_path: str, num_servers: int, config: Config, **_kwargs):
    """Run the same local orchestrator flow as the old argparse entrypoint."""
    cfg = parse_yaml_file_as(Config, config_path)
    base_dict = cfg.model_dump(exclude_unset=True)
    override_dict = config.model_dump(exclude_unset=True)
    merged_dict = always_merger.merge(base_dict, override_dict)
    merged_cfg = cfg.model_validate(merged_dict)

    resolved_public_ip = public_ip or get_public_ip()
    print("Public IP:", resolved_public_ip)

    fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="distqat_config_")
    os.close(fd)

    try:
        with open(temp_config_path, "w") as temp_config_file:
            yaml.dump(merged_cfg.model_dump(mode="json"), temp_config_file)

        async def run():
            orch = Orchestrator(config_path=temp_config_path, public_ip=resolved_public_ip, num_servers=num_servers)
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
