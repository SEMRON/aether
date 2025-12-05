from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from pathlib import Path
import json

class Server(BaseModel):
    hostname: str
    ssh_port: int = 22
    username: str
    key_path: Optional[str] = None
    remote_root_dir: str = "~/distqat"
    
    # Optional display name
    name: Optional[str] = None
    
    # Network/Process configuration
    mapped_monitor_port: Optional[int] = None
    mapped_host_port: Optional[int] = None
    grpc_announce_port: Optional[int] = None
    
    # Worker configuration
    num_servers: int = 1
    device: str = "cpu"
    batch_size: int = 16
    inner_steps: int = 50

    @property
    def display_name(self) -> str:
        return self.name or self.hostname

class JobConfig(BaseModel):
    config_path: str = "configs/resnet18.yaml"
    wandb_api_key: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    hf_token: Optional[str] = None

class GlobalState(BaseModel):
    servers: List[Server] = Field(default_factory=list)
    last_job_config: JobConfig = Field(default_factory=JobConfig)
    
    # Runtime state (not necessarily persisted to disk in this model, 
    # but helpful to have structure if we want to save session)
    
    def save_to_file(self, path: str = "gui_state.json"):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_file(cls, path: str = "gui_state.json") -> "GlobalState":
        if Path(path).exists():
            with open(path, "r") as f:
                try:
                    return cls.model_validate_json(f.read())
                except Exception:
                    pass
        return cls()

