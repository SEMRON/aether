import asyncio
import os
import signal
import subprocess
from pathlib import Path
from typing import Dict, Optional
import secrets
import string

from pydantic_yaml import parse_yaml_file_as
from distqat.config import Config


class BaseOrchestrator:
    """Base orchestrator class with common functionality for managing distributed processes."""
    
    def __init__(
        self,
        config_path: str,
        public_ip: Optional[str] = None,
        disable_quant: bool = True,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config_path: Path to the configuration YAML file
            public_ip: Public IP address for network configuration
            disable_quant: Whether to disable quantization
        """
        self.config = parse_yaml_file_as(Config, config_path)
        self.config_path = config_path
        self.disable_quant = disable_quant
        self.public_ip = public_ip
        
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_run_id = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))

        # Process tracking
        self.server_procs: Dict[str, subprocess.Popen] = {}
        self.trainer_procs: Dict[str, subprocess.Popen] = {}
        self.monitor_proc: Optional[subprocess.Popen] = None
        self.client_proc: Optional[subprocess.Popen] = None
        self.baseline_model_trainer_proc: Optional[subprocess.Popen] = None

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

