from typing import Callable, Iterable, Tuple, Type

import torch
from torch.optim import Optimizer as TorchOptimizer

from distqat.distributed.optim.collaborative import CollaborativeOptimizer
from distqat.distributed.optim.diloco import DiLoCoOptimizer
from distqat.config import OptimConfig, DilocoConfig
from distqat.models.biggan.biggan_adapter import InnerGANOptimizer

OptimizerFactory = Callable[..., torch.optim.Optimizer]


def get_optimizer_factory(config: OptimConfig) -> OptimizerFactory:
    if config.type == "adam":
        return lambda *args,**kwargs: torch.optim.AdamW(
            *args, **kwargs,
            lr=config.adam_lr,
            weight_decay=config.adam_weight_decay,
            betas=(config.adam_betas1, config.adam_betas2),
            eps=config.adam_epsilon,
        )
    elif config.type == "sgd":
        return lambda *args, **kwargs: torch.optim.SGD(
            *args, **kwargs,
            lr=config.sgd_lr,
            momentum=config.sgd_momentum,
            nesterov=config.sgd_nesterov,
        )
    elif config.type == "biggan":
        def factory(*, params, expert, **kwargs):
            params = list(params)
            G_set = set(expert.G.parameters())
            D_set = set(expert.D.parameters())
            G_params = [p for p in params if p in G_set]
            D_params = [p for p in params if p in D_set]
            param_groups = [{"params": G_params, "role": "G"}, {"params": D_params, "role": "D"}]
            return InnerGANOptimizer(
                param_groups=param_groups,
                g_opt_ctor=torch.optim.Adam,  # or AdamW
                g_opt_kwargs=dict(
                    lr=config.biggan_G_lr,
                    betas=(config.biggan_G_B1, config.biggan_G_B2),
                    eps=config.biggan_adam_eps,
                    # weight_decay=config.biggan_G_wd,
                ),
                d_opt_ctor=torch.optim.Adam,
                d_opt_kwargs=dict(
                    lr=config.biggan_D_lr,
                    betas=(config.biggan_D_B1, config.biggan_D_B2),
                    eps=config.biggan_adam_eps,
                    # weight_decay=getattr(config, "biggan_D_wd", 0.0),
                ),
            )
        return factory
    else:
        raise ValueError(f"Optimizer {config.type} not found")


def get_regular_optimizer_factory(config: DilocoConfig) -> OptimizerFactory:
    """
    Create a *regular* (non-DiLoCo) optimizer factory.

    This intentionally uses only the DiLoCo config's `inner_optim` section as a standalone optimizer,
    i.e. it does not wrap it in DiLoCo's inner/outer collaborative logic.
    """
    return get_optimizer_factory(config.inner_optim)


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
        target_group_size=config.target_group_size,
        min_group_size=config.min_group_size,
        min_matchmaking_time=config.min_matchmaking_time,
        request_timeout=config.request_timeout,
    )

def get_diloco_optimizer_cls_kwargs(run_id: int, config: DilocoConfig) -> Tuple[Type[TorchOptimizer], dict]:
    return DiLoCoOptimizer, dict(
        run_id=run_id,
        start=True,
        outer_optimizer=get_optimizer_factory(config.outer_optim),
        inner_optimizer=get_optimizer_factory(config.inner_optim),
        num_inner_steps=config.inner_steps,
        batch_size_per_step=config.batch_size_per_step,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        target_group_size=config.target_group_size,
        min_group_size=config.min_group_size,
        min_matchmaking_time=config.min_matchmaking_time,
        request_timeout=config.request_timeout,
    )

def get_regular_optimizer_cls_kwargs(config: OptimConfig) -> Tuple[Type[TorchOptimizer], dict]:
    """
    Return a torch optimizer class + kwargs for a *plain* (non-DiLoCo) optimizer.

    Note: for optimizers that require custom construction (e.g. BigGAN's InnerGANOptimizer),
    use `get_optimizer_factory()` / `get_regular_optimizer_factory()` instead.
    """
    if config.type == "adam":
        return torch.optim.AdamW, dict(
            lr=config.adam_lr,
            weight_decay=config.adam_weight_decay,
            betas=(config.adam_betas1, config.adam_betas2),
            eps=config.adam_epsilon,
        )
    if config.type == "sgd":
        return torch.optim.SGD, dict(
            lr=config.sgd_lr,
            momentum=config.sgd_momentum,
            nesterov=config.sgd_nesterov,
        )
    raise ValueError(
        f"Regular optimizer class/kwargs are not supported for optimizer type={config.type!r}. "
        f"Use get_optimizer_factory() instead."
    )