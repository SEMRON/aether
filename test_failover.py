import asyncio
import sys
import subprocess
import os
import signal
import time
import argparse

from hivemind.utils.logging import get_logger

from orchestrator import BaseOrchestrator
from run_script_utils import run_client_proc, run_server_proc, create_initial_peers_file, wait_for_initial_peers, run_monitor_proc

logger = get_logger(__name__)

DISABLE_QUANT = True
NUM_SERVERS = 2

class Orchestrator(BaseOrchestrator):
    def __init__(self, config_path: str, public_ip=None):
        super().__init__(
            config_path=config_path,
            public_ip=public_ip,
            disable_quant=DISABLE_QUANT,
        )

    async def start(self):
        print("ORCHESTRATOR: Starting")

        initial_peers_path = create_initial_peers_file(log_dir=self.config.log_dir)

        # Start monitor
        self.monitor_proc = run_monitor_proc(config_path=self.config_path, refresh_period=2, store_ip_addresses_path=str(initial_peers_path), public_ip=self.public_ip, wandb_run_id=self.wandb_run_id)

        initial_peers_json = wait_for_initial_peers(initial_peers_path)

        # Spawn client, which in turn spawns trainers dynamically based on complete pipelines
        self.client_proc = run_client_proc(config_path=self.config_path, refresh_period=5, network_initial_peers=initial_peers_json, public_ip=self.public_ip, disable_quant=self.disable_quant)

        # Spawn servers
        for expert_idx in range(NUM_SERVERS):                
            for stage_idx,  pipeline_step_cfg in enumerate(self.config.model_pipeline.pipeline):
                _, stage = pipeline_step_cfg.model_name.split(".")
                # set expert and stage index to None to auto-discover
                self.server_procs[f"server_{stage}_{expert_idx}"] = run_server_proc(config_path=self.config_path, network_initial_peers=initial_peers_json, public_ip=self.public_ip, idx=expert_idx, stage_index=stage_idx, disable_quant=self.disable_quant)
            
                # Give servers time to write their expert UID to the DHT
                time.sleep(3)
        

        # Test combination of shutdowns to see failover mechanism
        time.sleep(20)
        self.shutdown_server(expert_idx=0, stage_name="tail")
        time.sleep(5)
        self.shutdown_server(expert_idx=1, stage_name="head")

        time.sleep(120)
        self.shutdown_server(expert_idx=0, stage_name="head")
        time.sleep(10)
        self.server_procs[f"server_head_0"] = run_server_proc(config_path=self.config_path, network_initial_peers=initial_peers_json, public_ip=self.public_ip, idx=0, stage_index=0, disable_quant=self.disable_quant)


    def shutdown_server(self, expert_idx: int, stage_name: str):
        proc = self.server_procs[f"server_{stage_name}_{expert_idx}"]
        if proc is not None and proc.poll() is None:
            os.kill(proc.pid, signal.SIGINT)
        time.sleep(1)

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

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-ip", type=str, default=None)
    parser.add_argument("--config-path", type=str, default="configs/resnet18_split.yaml")
    args = parser.parse_args()

    orch = Orchestrator(config_path=args.config_path, public_ip=args.public_ip)
    try:
        await orch.start()
        await orch.wait()

        print("Training finished. Shutting down...")
    except KeyboardInterrupt:
        print("Keyboard interrupt. Shutting down...")
    finally:
        await orch.shutdown()

    print("Done")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        print("Done")
