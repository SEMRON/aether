from typing import Callable, Iterable, Tuple, Type

import torch
from torch.optim import Optimizer as TorchOptimizer

from distqat.distributed.optim.collaborative import CollaborativeOptimizer
from distqat.distributed.optim.diloco import DiLoCoOptimizer
from distqat.config import OptimConfig, DilocoConfig

def get_optimizer_factory(config: OptimConfig) -> Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]:
    if config.type == "adam":
        return lambda *args,**kwargs: torch.optim.AdamW(
            *args, **kwargs,
            lr=config.adam_lr,
            weight_decay=config.adam_weight_decay,
            betas=(config.adam_betas1, config.adam_betas2),
        )
    elif config.type == "sgd":
        return lambda *args, **kwargs: torch.optim.SGD(
            *args, **kwargs,
            lr=config.sgd_lr,
            momentum=config.sgd_momentum,
            nesterov=config.sgd_nesterov,
        )
    else:
        raise ValueError(f"Optimizer {config.type} not found")

def get_collaborative_optimizer_cls_kwargs(run_id: int, config: DilocoConfig) -> Tuple[Type[TorchOptimizer], dict]:
    return CollaborativeOptimizer, dict(
        run_id=run_id,
        start=True,
        optimizer=get_optimizer_factory(config.inner_optim),
        target_batch_size=config.inner_steps * config.batch_size_per_step,
        batch_size_per_step=config.batch_size_per_step,
        min_refresh_period=config.min_refresh_period,
        max_refresh_period=config.max_refresh_period,
        default_refresh_period=config.default_refresh_period,
        expected_drift_peers=config.expected_drift_peers,
        expected_drift_rate=config.expected_drift_rate,
        performance_ema_alpha=config.performance_ema_alpha,
        metadata_expiration=config.metadata_expiration,
        averaging_timeout=config.averaging_timeout,
        load_state_timeout=config.load_state_timeout,
        verbose=config.verbose,
    )

def get_diloco_optimizer_cls_kwargs(run_id: int, config: DilocoConfig) -> Tuple[Type[TorchOptimizer], dict]:
    return DiLoCoOptimizer, dict(
        run_id=run_id,
        start=True,
        outer_optimizer=get_optimizer_factory(config.outer_optim),
        inner_optimizer=get_optimizer_factory(config.inner_optim),
        num_inner_steps=config.inner_steps,
        batch_size_per_step=config.batch_size_per_step,
        min_refresh_period=config.min_refresh_period,
        max_refresh_period=config.max_refresh_period,
        default_refresh_period=config.default_refresh_period,
        expected_drift_peers=config.expected_drift_peers,
        expected_drift_rate=config.expected_drift_rate,
        performance_ema_alpha=config.performance_ema_alpha,
        metadata_expiration=config.metadata_expiration,
        averaging_timeout=config.averaging_timeout,
        load_state_timeout=config.load_state_timeout,
        verbose=config.verbose,
    )