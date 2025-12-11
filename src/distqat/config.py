from typing import Optional, Literal, List, Callable, Dict, Any
from typing import List, Optional

import json
import click
from pydantic import BaseModel, Field, PositiveInt, PrivateAttr, computed_field
from pydantic import field_validator
from pydanclick import from_pydantic
from pydantic_yaml import parse_yaml_file_as
from deepmerge import always_merger
import importlib
import os
from pathlib import Path
from .quant.scheme import QuantScheme
from .utils.rules import OverrideRule
from .utils.resolve import CompiledResolver, compile_overrides



class DataConfig(BaseModel):
    dataset_name: str = "mnist"
    dataset_config: Optional[str] = None
    dataset_split: Optional[str] = "train"
    hf_token: Optional[str] = None

    @field_validator("hf_token", mode="before")
    @classmethod
    def _fill_hf_token_from_env(cls, value: Optional[str]) -> Optional[str]:
        """
        Allow HF token to come from the HF_TOKEN environment variable when the
        value is not explicitly provided (in YAML or CLI).
        """
        if value is None:
            return os.getenv("HF_TOKEN")
        return value

    num_workers: int = 1
    shuffle_buffer_size: int = 8192

    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed"
    task_type: Literal["cv", "llm" , "speech", "image_gen", "node_pred", "rl"] = "cv"
    full_model_name: str = "EleutherAI/gpt-neo-1.3B"
    
    # CV-specific
    num_channels: int = 3
    img_size: int = 32

    # LLM-specific
    seq_len: int = 128
    
    # Speech-specific
    sampling_rate: int = 16000


class OptimConfig(BaseModel):
    type: Literal["sgd", "adam", "biggan"] = "sgd"
    
    # Parameters for SGD
    sgd_lr: float = 0.7
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True

    # Parameters for Adam
    adam_lr: float = 4e-4
    adam_weight_decay: float = 0.1
    adam_betas1: float = 0.9
    adam_betas2: float = 0.95

    # Parameters for BigGAN
    biggan_G_lr: float = 2e-4
    biggan_D_lr: float = 2e-4
    biggan_G_B1: float = 0.0
    biggan_D_B1: float = 0.0
    biggan_G_B2: float = 0.999
    biggan_D_B2: float = 0.999
    biggan_adam_eps: float = 1.0e-06
    biggan_model_config: Dict[str, Any] = {}

class DilocoConfig(BaseModel):
    inner_optim: OptimConfig = OptimConfig(type="adam")
    outer_optim: OptimConfig = OptimConfig(type="sgd")
    inner_steps: int = 50
    outer_steps: int = 10
    batch_size_per_step: int = 64
    gradient_accumulation_steps: int = 1
    min_refresh_period: float = 0.5
    max_refresh_period: float = 30
    default_refresh_period: float = 3
    expected_drift_peers: float = 3
    expected_drift_rate: float = 0.2
    performance_ema_alpha: float = 0.1
    metadata_expiration: float = 60.0
    averaging_timeout: Optional[float] = None
    load_state_timeout: float = 600.0
    verbose: bool = True

    @computed_field
    @property
    def total_steps(self) -> int:
        return self.inner_steps * self.outer_steps

class ConcreteQuantConfigEntry(BaseModel):
    weight_quant: QuantScheme = Field(
        default_factory=lambda: QuantScheme(
            algo="LearnedStepQuantizer", num_bits=8, slice_bits=4
        )
    )
    activation_quant: QuantScheme = Field(
        default_factory=lambda: QuantScheme(
            algo="LearnedStepQuantizer", num_bits=8, slice_bits=1
        )
    )
    accumulator_length: PositiveInt = 512
    accumulator_bits: PositiveInt = 8


class PartialQuantConfigEntry(BaseModel):
    """
    Partial overlay: any None field is treated as "inherit".
    """

    weight_quant: Optional[QuantScheme] = None
    activation_quant: Optional[QuantScheme] = None

    accumulator_length: Optional[PositiveInt] = None
    accumulator_bits: Optional[PositiveInt] = None


class QuantConfig(BaseModel):
    """
    Top-level config with base values and override rules.
    Compile once to get a resolver; it's cached on the instance.
    """

    base: ConcreteQuantConfigEntry = Field(default_factory=ConcreteQuantConfigEntry)
    overrides: List[OverrideRule[PartialQuantConfigEntry]] = Field(default_factory=list)

    _compiled: Optional[
        CompiledResolver[ConcreteQuantConfigEntry, PartialQuantConfigEntry]
    ] = PrivateAttr(default=None)

    def compile(
        self, *, force: bool = False
    ) -> CompiledResolver[ConcreteQuantConfigEntry, PartialQuantConfigEntry]:
        """
        Build (or rebuild) the compiled resolver. Call with force=True to rebuild after mutating overrides.
        """
        if self._compiled is None or force:
            self._compiled = compile_overrides(self.base, self.overrides)
        return self._compiled

    def resolve_for(
        self, module_path: str, module_cls_name: str
    ) -> ConcreteQuantConfigEntry:
        """
        Convenience passthrough so callers don't have to hold a resolver explicitly.
        """
        return self.compile().resolve_for(module_path, module_cls_name)



class ModelConfig(BaseModel):
    model_name: str = "mlp.full"
    num_classes: int = 10
    hid_dim: int = 2048
    n_layers: int = 8
    idx: int = 8
    # Arbitrary model-specific parameters forwarded to expert constructor and sample_input
    extra: dict = Field(default_factory=dict)

class ModelPipelineConfig(BaseModel):
    pipeline: List[ModelConfig] = [ModelConfig()]
    forward_timeout: float = 40.0
    backward_timeout: float = 90.0

class NetworkConfig(BaseModel):
    """Configuration for multi-peer networking"""
    initial_peers: List[str] = []
    host_maddrs: List[str] = ["/ip4/0.0.0.0/tcp/0"]
    announce_maddrs: List[str] = []
    use_ipfs: bool = False
    client_mode: bool = False
    identity_path: str = "peer_key"
    hivemind_compression: Literal["none", "fp16", "scaled-fp16", "uniform8bit", "quantile8bit", "blockwise8bit"] | None = None
    skip_load_from_peers: bool = False

    # Ports configuration
    monitor_port: int = 50000
    client_port: int = 51000
    trainer_base_port: int = 50100 # Base port for trainer, actual port = base + idx (0 to max_expert_index)
    server_base_hostport: int = 50500  # Base port for server host (local binding), actual port = base + idx (0 to num_servers)
    server_base_hostport_announce: int = 50500  # Base port for server host announce (DHT peer discovery), necessary when port mapping is used, actual port = base + idx (0 to num_servers)
    server_base_grpcport: int = 51500  # Base port for server gRPC (local binding), actual port = base + idx (0 to num_servers)
    server_base_grpc_announceport: int = 51500  # Base port for server gRPC announce (expert endpoint), necessary when port mapping is used, actual port = base + idx (0 to num_servers)

class ParamMirrorConfig(BaseModel):
    enable: bool = True
    refresh_every: int = 300


class DataServerConfig(BaseModel):
    ipc_host: str = "127.0.0.1"
    ipc_port: int = 52555
    ipc_key: str = "distqat"
    num_workers: int = 2
    pool_size: int = 32

class Config(BaseModel):
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    experiment_prefix: str = "default_experiment"
    wandb_run_id: Optional[str] = None # This is required for logging from multiple processes
    data: DataConfig = DataConfig()
    diloco: DilocoConfig = DilocoConfig()
    model_pipeline: ModelPipelineConfig = ModelPipelineConfig()
    quant: QuantConfig = QuantConfig()
    network: NetworkConfig = NetworkConfig()
    max_expert_index: int = 128
    device: str = "cuda"
    param_mirror: ParamMirrorConfig = ParamMirrorConfig()
    data_server: DataServerConfig = DataServerConfig()
    checkpoint_dir: Optional[Path] = None
    log_dir: Path = Path("logs") / experiment_prefix

    # Optional holder for model-specific configs (e.g., BigGAN CLI-style params)
    biggan: Optional[Dict[str, Any]] = None

    # Attribute to store the config path
    path: Optional[str] = None

    @field_validator("wandb_project", mode="before")
    @classmethod
    def _normalize_wandb_project(cls, value):
        if isinstance(value, str) and value.strip().lower() == "none":
            return None
        return value

@click.command()
@click.option("--config-path", type=str, default=None)
@click.option("--network-initial-peers", type=str, default=None, help="JSON array string or comma-separated list of initial peers")
@from_pydantic(Config, exclude=["network.initial_peers"])
def parse_args(config_path: Optional[str], network_initial_peers: Optional[str], config: Config, **extra_kwargs):
    if config_path is None:
        cfg = Config()
    else:
        cfg = parse_yaml_file_as(Config, config_path)

    base_dict = cfg.model_dump(exclude_unset=True)
    nxt_dict = config.model_dump(exclude_unset=True)
    merged_dict = always_merger.merge(base_dict, nxt_dict)

    merged_cfg = cfg.model_validate(merged_dict)
    
    # Handle network-initial-peers manually (try JSON first, then comma-separated)
    if network_initial_peers is not None:
        try:
            parsed_json = json.loads(network_initial_peers)
            if isinstance(parsed_json, list):
                merged_cfg.network.initial_peers = parsed_json
            else:
                merged_cfg.network.initial_peers = [p.strip() for p in network_initial_peers.split(",") if p.strip()]
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse network-initial-peers: {e}")
    
    if merged_cfg.device == "rocm":
        merged_cfg.device = "cuda"

    # Attach the config path to the config object for downstream Trainers (spawned from the client)
    merged_cfg.path = config_path

    return merged_cfg, extra_kwargs
