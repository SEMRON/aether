from typing import List, Optional, Dict
import warnings
import time
import click
import subprocess
import sys
import json
import os
import signal
from pathlib import Path

import torch
torch.multiprocessing.set_sharing_strategy('file_descriptor')

from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.dht import DHT
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import SchemaValidator
from hivemind.optim.progress_tracker import TrainingProgressSchema
import wandb
from distqat.config import Config
from distqat.config import parse_args
from distqat.distributed.utils.auto_discovery import discover_experts
from distqat.distributed.utils.reassignment_monitor import store_reassignment_signal
from distqat.distributed.param_mirror import ParamMirror
from distqat.utils import logging


logger = get_logger(__name__)
use_hivemind_log_handler("in_root_logger")

class SwarmClient:
    def __init__(
        self, 
        config: Config, 
        public_ip: Optional[str] = None,
        refresh_period: int = 300,
        disable_quant: bool = False,
    ):
        self.config = config
        self.refresh_period = refresh_period
        self.trainer_procs: Dict[int, subprocess.Popen] = {}
        self.log_dir = config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.disable_quant = disable_quant
        self.wandb_enabled = config.wandb_project is not None
        
        self.dht = DHT(
            start=True,
            initial_peers=config.network.initial_peers,
            host_maddrs=config.network.host_maddrs,
            announce_maddrs=config.network.announce_maddrs,
        )

        logger.info(f"CLIENT: Visible multiaddresses: {self.dht.get_visible_maddrs(latest=True)}")

        signature_validator = RSASignatureValidator()
        self.dht.add_validators([SchemaValidator(TrainingProgressSchema, prefix=config.experiment_prefix), signature_validator])

        if self.wandb_enabled:
            dht_run_id = logging.get_wandb_run_id(self.dht, config.experiment_prefix)
            if dht_run_id:
                config.wandb_run_id = dht_run_id
                logger.info(f"CLIENT:Retrieved wandb_run_id from DHT: {dht_run_id}")
            else:
                logger.warning("CLIENT:wandb_run_id not found in DHT and not provided in config. Wandb may create separate runs.")
        

        if self.wandb_enabled:
            try:
                wandb_kwargs = dict(
                    entity=config.wandb_entity,
                    project=config.wandb_project,
                    name=f"{config.experiment_prefix}",
                )
                if config.wandb_run_id:
                    wandb_kwargs["id"] = config.wandb_run_id
                    wandb_kwargs["resume"] = "allow"
                wandb.init(**wandb_kwargs)
            except Exception as e:
                logger.warning(f"CLIENT: Failed to initialize wandb for client: {e}. Continuing without wandb logging.")
                self.wandb_enabled = False

        self.trainer_base_port = config.network.trainer_base_port
        self.public_ip = public_ip

        self.done = False

        # Fallback parameter mirror
        self._param_mirror: Optional[ParamMirror] = None
        if self.config.param_mirror.enable:
            try:
                self._param_mirror = ParamMirror(
                    self.config, self.dht,
                    refresh_every=self.config.param_mirror.refresh_every,
                )
                self._param_mirror.start()
                logger.info("Parameter mirror started")
            except Exception as e:
                logger.warning(f"Failed to start parameter mirror: {e}")


        # Start data server
        self.data_server_proc = subprocess.Popen(
            [
                sys.executable, "src/distqat/distributed/data_server.py",
                "--config-path", self.config.path,
            ],
            stdout=open(self.log_dir / "data_server.log", "w"),
            stderr=subprocess.STDOUT,
            text=True,
        )

        logging.setup_file_logging(self.log_dir / "client.log", wandb_enabled=self.wandb_enabled)
        logging.track_log_file(self.log_dir / "data_server.log", self.wandb_enabled)
        


    def _get_pipeline_batch_size_and_inner_steps(self, pipeline_info: Dict) -> Optional[int]:
        """
        Determine the batch_size_per_step and inner_steps for a pipeline.
        
        If all stages have the same batch_size_per_step and inner_steps, use that.
        Otherwise, use the first non-None value found, or fall back to config default.
        """
        batch_sizes = []
        inner_steps = []
        for stage_name, stage_info in pipeline_info.items():
            batch_size = stage_info.get('batch_size_per_step')
            inner_step = stage_info.get('inner_steps')
            if batch_size is not None:
                batch_sizes.append(batch_size)
            if inner_step is not None:
                inner_steps.append(inner_step)
        
        if not batch_sizes:
            logger.warning("No batch_size_per_step found for pipeline stages, using config default")
            return self.config.diloco.batch_size_per_step, self.config.diloco.inner_steps
        
        # Check if all stages have the same batch size
        unique_batch_sizes = set(batch_sizes)
        unique_inner_steps = set(inner_steps)
        if len(unique_batch_sizes) == 1 and len(unique_inner_steps) == 1    :
            return batch_sizes[0], inner_steps[0]
        else:
            logger.warning(
                f"Pipeline stages have different batch sizes: {unique_batch_sizes} and inner steps: {unique_inner_steps}. "
                f"Using first value: {batch_sizes[0]} and {inner_steps[0]}"
            )
            return batch_sizes[0], inner_steps[0]

    def spawn_trainer(self, trainer_id: int, initial_peers: List[str], batch_size_per_step: Optional[int] = None, inner_steps: Optional[int] = None):
        """Spawn a trainer process for the given trainer_id"""
        if trainer_id in self.trainer_procs:
            logger.warning(f"Trainer {trainer_id} already exists, skipping spawn")
            return

        initial_peers_json = json.dumps(initial_peers)
        
        cmd = [
            sys.executable, "src/distqat/distributed/trainer.py",
            "--trainer-id", str(trainer_id),
            "--config-path", self.config.path,
            "--network-initial-peers", initial_peers_json,
        ]

        if self.public_ip:
            hostmaddr, announcemaddr = f'["/ip4/0.0.0.0/tcp/{self.trainer_base_port + trainer_id}"]', f'["/ip4/{self.public_ip}/tcp/{self.trainer_base_port + trainer_id}", "/ip4/127.0.0.1/tcp/{self.trainer_base_port + trainer_id}"]'
            cmd.extend(["--network-host-maddrs", hostmaddr])
            cmd.extend(["--network-announce-maddrs", announcemaddr])

        if batch_size_per_step is not None:
            cmd.extend(["--diloco-batch-size-per-step", str(batch_size_per_step)])
        
        if inner_steps is not None:
            cmd.extend(["--diloco-inner-steps", str(inner_steps)])

        if self.disable_quant:
            cmd.append("--disable-quant")
        
        log_path = self.log_dir / f"trainer_{trainer_id}.log"
        log_file = open(log_path, "w")
        
        try:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
            self.trainer_procs[trainer_id] = proc
            logger.info(f"Spawned trainer {trainer_id} with PID {proc.pid}")
            logging.track_log_file(log_path)
        except Exception as e:
            logger.error(f"Failed to spawn trainer {trainer_id}: {e}")
            log_file.close()

    def stop_trainer(self, trainer_id: int):
        """Stop a trainer process"""
        if trainer_id not in self.trainer_procs:
            logger.warning(f"Trainer {trainer_id} not found, cannot stop")
            return
            
        proc = self.trainer_procs[trainer_id]
        if proc.poll() is None:
            logger.info(f"Stopping trainer {trainer_id} (PID {proc.pid})")
            os.kill(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
                logger.info(f"Trainer {trainer_id} stopped successfully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Trainer {trainer_id} did not stop gracefully, forcing termination")
                os.kill(proc.pid, signal.SIGKILL)
        
        del self.trainer_procs[trainer_id]

    def reassign_incomplete_experts(self, incomplete_pipelines: Dict[int, Dict]):
        """Send signal to DHT so that servers of incomplete pipelines will restart and join the swarm as full pipelines again """
        if len(incomplete_pipelines) == 0:
            return
        # Determine if we have enough total idle servers (across all incomplete pipelines)
        pipeline_len = len(self.config.model_pipeline.pipeline)
        num_idle_servers = sum(len(info.get('stages', {})) for info in incomplete_pipelines.values())
        if num_idle_servers < pipeline_len:
            logger.info(
                f"Not reassigning incomplete experts: only {num_idle_servers} idle servers available, "
                f"need >= {pipeline_len} to form at least one complete pipeline"
            )
            return

        for expert_id, info in incomplete_pipelines.items():
            logger.info(f"Reassigning incomplete expert {expert_id} to new trainer")
            logger.info(f"Info: {info}")
            for stage, stage_info in info['stages'].items():
                expert_uid = stage_info['uid']
                store_reassignment_signal(self.dht, expert_uid)

    def manage_trainers(self, complete_pipelines: Dict[int, Dict]):
        """Manage trainer processes based on discovered complete pipelines"""
        current_trainer_ids = set(self.trainer_procs.keys())
        expected_trainer_ids = set(complete_pipelines.keys())
        
        # Get initial peers from DHT
        initial_peers = [str(addr) for addr in self.dht.get_visible_maddrs(latest=True)]
        
        # Start new trainers for new complete pipelines
        for trainer_id in expected_trainer_ids - current_trainer_ids:
            logger.info(f"Starting trainer for complete pipeline {trainer_id}")
            # Determine batch_size_per_step for this pipeline
            pipeline_info = complete_pipelines[trainer_id]
            batch_size, inner_steps = self._get_pipeline_batch_size_and_inner_steps(pipeline_info)
            logger.info(f"Pipeline {trainer_id} will use batch_size_per_step={batch_size} and inner_steps={inner_steps}")
            self.spawn_trainer(trainer_id, initial_peers, batch_size_per_step=batch_size, inner_steps=inner_steps)
        
        # Stop trainers for pipelines that are no longer complete
        for trainer_id in current_trainer_ids - expected_trainer_ids:
            logger.info(f"Stopping trainer for incomplete pipeline {trainer_id}")
            self.stop_trainer(trainer_id)
        
        # Check for crashed trainers and restart them
        crashed_trainers = []
        completed_trainers = []
        for trainer_id, proc in self.trainer_procs.items():
            if proc.poll() is not None:
                if proc.returncode == 0:
                    completed_trainers.append(trainer_id)
                    logger.info(f"Trainer {trainer_id} completed training successfully")
                else:
                    crashed_trainers.append(trainer_id)
                    logger.warning(f"Trainer {trainer_id} crashed with return code {proc.returncode}")
        
        for trainer_id in crashed_trainers:
            logger.info(f"Restarting crashed trainer {trainer_id}")
            # Get batch_size_per_step for this pipeline if it exists
            batch_size = None
            if trainer_id in complete_pipelines:
                pipeline_info = complete_pipelines[trainer_id]
                batch_size, inner_steps = self._get_pipeline_batch_size_and_inner_steps(pipeline_info)
            self.stop_trainer(trainer_id)
            self.spawn_trainer(trainer_id, initial_peers, batch_size_per_step=batch_size, inner_steps=inner_steps)


        
        if len(completed_trainers) > 0 and len(completed_trainers) == len(expected_trainer_ids):
            logger.info("All training completed - shutting down client")
            self.shutdown()
            self.done = True

    def run(self):
        dht = self.dht
        config = self.config

        print("CLIENT: Visible multiaddresses:", dht.get_visible_maddrs(latest=True))
        while not self.done:
            # Use the enhanced dht_handler method to get both complete and incomplete pipelines
            logger.info("=== Expert Discovery ===")
            
            complete_pipelines, incomplete_pipelines = discover_experts(dht, config)

            logger.info(f"Expert Discovery Summary:")
            logger.info(f"  Complete pipelines: {len(complete_pipelines)} - {list(complete_pipelines.keys())}")
            logger.info(f"  Incomplete pipelines: {len(incomplete_pipelines)} - {list(incomplete_pipelines.keys())}")
            
            # Log details about incomplete pipelines
            for expert_id, info in incomplete_pipelines.items():
                logger.info(f"  Expert {expert_id}: has {list(info['stages'].keys())}, missing {info['missing_stages']}")

            # Manage trainer processes based on discovered pipelines
            self.manage_trainers(complete_pipelines)

            # Reassign incomplete experts 
            self.reassign_incomplete_experts(incomplete_pipelines)

            time.sleep(self.refresh_period)
    
    def shutdown(self):
        # Stop all trainer processes
        logger.info("Shutting down all trainer processes...")
        for trainer_id in list(self.trainer_procs.keys()):
            self.stop_trainer(trainer_id)

        # Stop data server and wait to ensure the port is freed
        if getattr(self, "data_server_proc", None) and self.data_server_proc.poll() is None:
            try:
                os.kill(self.data_server_proc.pid, signal.SIGINT)
                self.data_server_proc.wait(timeout=10.0)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                logger.warning("Data server did not stop gracefully, forcing termination")
                try:
                    os.kill(self.data_server_proc.pid, signal.SIGKILL)
                except OSError:
                    pass
        
        if self._param_mirror is not None:
            self._param_mirror.stop()

        self.dht.shutdown()
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb for client: {e}")

def run_client(cfg: Config, refresh_period: int, public_ip: Optional[str] = None, disable_quant: bool = False):
    client = SwarmClient(
        cfg,
        refresh_period=refresh_period,
        public_ip=public_ip,
        disable_quant=disable_quant,
    )
    
    logger.info(f"Experiment prefix: {cfg.experiment_prefix}")
    logger.info(f"Refresh period: {refresh_period} seconds")
    
    try:
        client.run()
    except KeyboardInterrupt:
        logger.info("Client interrupted, shutting down...")
    finally:
        client.shutdown()


if __name__ == "__main__":
    parse_args_with_extra_kwargs = click.option("--refresh-period", type=int, default=5)(parse_args)
    parse_args_with_extra_kwargs = click.option("--disable-quant", is_flag=True)(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option("--public-ip", type=str, default=None)(parse_args_with_extra_kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*is used more than once. Remove its duplicate as parameters should be unique.*")
        cfg, extra_kwargs = parse_args_with_extra_kwargs(standalone_mode=False)
        run_client(
            cfg,
            **extra_kwargs
        )
