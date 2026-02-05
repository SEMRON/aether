import asyncio
import os
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import secrets
import string

from hivemind.dht import DHT
from pydantic_yaml import parse_yaml_file_as
from distqat.config import Config
from distqat.utils.logging import get_wandb_run_id_with_retries, store_wandb_run_id


class BaseOrchestrator:
    """Base orchestrator class with common functionality for managing distributed processes."""
    
    def __init__(
        self,
        config_path: str,
        public_ip: Optional[str] = None,
        initial_peers: Optional[List[str]] = None,
        is_monitor: bool = False,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config_path: Path to the configuration YAML file
            public_ip: Public IP address for network configuration
            disable_quant: Whether to disable quantization
            initial_peers: List of initial DHT peers to connect to for retrieving existing wandb_run_id
            is_monitor: If True, this orchestrator can generate new wandb run IDs. Workers should set this to False.
        """
        self.config = parse_yaml_file_as(Config, config_path)
        self.config_path = config_path
        self.public_ip = public_ip
        
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Try to retrieve existing wandb_run_id from DHT to preserve run continuity across restarts
        # Only the monitor should generate new IDs; workers should use what's in DHT
        self.wandb_run_id = self._get_or_create_wandb_run_id(initial_peers, is_monitor=is_monitor)

        # Process tracking
        self.server_procs: Dict[str, subprocess.Popen] = {}
        self.trainer_procs: Dict[str, subprocess.Popen] = {}
        self.monitor_proc: Optional[subprocess.Popen] = None
        self.client_proc: Optional[subprocess.Popen] = None
        self.baseline_model_trainer_proc: Optional[subprocess.Popen] = None
        self.evaluator_proc: Optional[subprocess.Popen] = None
        self.baseline_evaluator_proc: Optional[subprocess.Popen] = None

    def _get_or_create_wandb_run_id(self, initial_peers: Optional[List[str]], is_monitor: bool = False) -> Optional[str]:
        """
        Try to retrieve existing wandb_run_id from DHT, or generate a new one if this is the monitor.
        
        This ensures run continuity across restarts - if a wandb run was already
        created for this experiment, we'll resume it instead of creating a new one.
        
        Args:
            initial_peers: List of initial DHT peers to connect to
            is_monitor: If True, this is the monitor process and should generate a new ID if not found.
                        If False, this is a worker and should NOT generate a new ID to avoid creating
                        separate wandb runs.
            
        Returns:
            The wandb run ID (retrieved from DHT, newly generated for monitor, or None for workers)
        """
        wandb_run_id = None
        
        # Try to retrieve from DHT if we have peers to connect to
        if initial_peers and len(initial_peers) > 0:
            dht = None
            try:
                # Use full network config for DHT to ensure proper connectivity
                # (minimal DHT with only initial_peers may have issues retrieving values)
                dht = DHT(
                    start=True, 
                    initial_peers=initial_peers,
                    host_maddrs=self.config.network.host_maddrs,
                    announce_maddrs=self.config.network.announce_maddrs,
                )
                # Use retries since the monitor may not have stored the ID yet
                wandb_run_id = get_wandb_run_id_with_retries(
                    dht, self.config.experiment_prefix, max_retries=15, retry_delay=2.0
                )
                if wandb_run_id:
                    print(f"ORCHESTRATOR: Retrieved existing wandb_run_id from DHT: {wandb_run_id}")
                else:
                    print("ORCHESTRATOR: No existing wandb_run_id found in DHT after retries")
            except Exception as e:
                print(f"ORCHESTRATOR: Failed to connect to DHT for wandb_run_id lookup: {e}")
            finally:
                if dht is not None:
                    try:
                        dht.shutdown()
                    except Exception:
                        pass
        
        # Only generate new ID if this is the monitor process (no initial_peers means we're starting fresh)
        # Workers should NOT generate new IDs - they should use what the monitor stored in DHT
        if wandb_run_id is None and is_monitor:
            wandb_run_id = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
            print(f"ORCHESTRATOR: Generated new wandb_run_id: {wandb_run_id}")
        elif wandb_run_id is None:
            print("ORCHESTRATOR: WARNING - No wandb_run_id found and not generating new one (not monitor). Wandb logging may be disabled or create separate runs.")
        
        return wandb_run_id

    async def start(self):
        """Start all processes. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement start()")

    async def wait(self):
        """Wait for processes to complete. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement wait()")

    def _get_all_processes(self) -> Dict[str, Optional[subprocess.Popen]]:
        """Get a dictionary of all tracked processes."""
        return {
            "monitor": self.monitor_proc,
            "client": self.client_proc,
            "baseline_trainer": self.baseline_model_trainer_proc,
            **self.trainer_procs,
            **self.server_procs,
            "evaluator": self.evaluator_proc,
            "baseline_evaluator": self.baseline_evaluator_proc,
        }

    async def shutdown(self):
        """Shutdown all processes gracefully."""
        print("ORCHESTRATOR: Shutting down")
        all_processes = self._get_all_processes()

        # First, send SIGINT to all running processes
        for p in all_processes.values():
            if p is not None and p.poll() is None:
                os.kill(p.pid, signal.SIGINT)

        # Wait for processes to terminate, with timeout
        while True:
            if all(p.poll() is not None for p in all_processes.values() if p is not None):
                break

            # If still running after initial SIGINT, send another SIGINT
            for label, p in all_processes.items():
                if p is not None and p.poll() is None:
                    print(f"ORCHESTRATOR: Terminating {label}")
                    os.kill(p.pid, signal.SIGINT)

            # Wait with timeout
            for label, p in all_processes.items():
                if p is not None:
                    try:
                        p.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        print(f"ORCHESTRATOR: {label} still running")

