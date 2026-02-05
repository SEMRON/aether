import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable, Dict, Iterable, Iterator, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from itertools import chain
from threading import Event, Lock
from typing import Dict, Iterator, Optional, Sequence

import numpy as np
import torch
from pydantic.v1 import BaseModel, StrictBool, StrictFloat, confloat, conint

from hivemind.dht import DHT
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.optim.optimizer import OptimizerFactory, TorchOptimizer
from hivemind.optim.training_averager import initialize_optimizer_state, dump_optimizer_state, load_optimizer_state
from hivemind.utils import get_dht_time, get_logger, PerformanceEMA
from hivemind.averaging import DecentralizedAverager
from hivemind.compression import CompressionInfo, TensorRole

from distqat.distributed.optim.base import DecentralizedOptimizerBase


logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)

@dataclass(frozen=False)
class CollaborationState:
    outer_step: int
    inner_steps_accumulated: int
    num_inner_steps: int
    num_peers: int
    num_clients: int
    eta_next_step: float
    next_fetch_time: float
    any_peer_ready: bool = False  # Tracks if any peer has completed inner steps

    def ready_for_outer_step(self, local_inner_step: int, min_local_steps: int):
        """
        Ready for outer step if:
        1. Local peer has done at least min_local_steps, AND
        2. Either local peer reached num_inner_steps OR another peer is already ready
        
        This allows the first peer to trigger averaging for all other peers.
        """
        local_done = local_inner_step + 1 >= self.num_inner_steps
        has_minimum_progress = local_inner_step >= min_local_steps
        
        return has_minimum_progress and (local_done or self.any_peer_ready)

    def register_step(self, outer_step: int):
        self.outer_step = max(outer_step, self.outer_step)
        self.inner_steps_accumulated = 0
        self.eta_next_step = float("inf")
        self.any_peer_ready = False  # Reset for next round

class TrainingState(BaseModel):
    peer_id: bytes
    outer_step: conint(ge=0, strict=True)
    inner_step: conint(ge=0, strict=True)
    inner_steps_per_second: confloat(ge=0.0, strict=True)
    time: StrictFloat
    client_mode: StrictBool


class TrainingProgressSchema(BaseModel):
    progress: Dict[BytesWithPublicKey, Optional[TrainingState]]


class DiLoCoTrainingAverager(DecentralizedAverager):
    """
    TODO
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        avg_only_params: Optional[Iterable[torch.Tensor]] = None,
        average_opt_statistics: Sequence[str] = (),
        extra_tensors: Sequence[torch.Tensor] = (),
        initialize_optimizer: bool = True,
        **kwargs,
    ):
        parameter_names = tuple(i for group in optimizer.param_groups for i in range(len(group["params"])))
        if avg_only_params is not None:
            self.avg_only_params_list = list(avg_only_params)
        else:
            self.avg_only_params_list = []
        
        self.opt, self.extra_tensors, self.outer_step = optimizer, tuple(extra_tensors), 0
        self.opt_statistics = tuple(average_opt_statistics)
        self.parameter_names = parameter_names
        self.step_executor = ThreadPoolExecutor(max_workers=1)
        self.lock_averager_step = Lock()
        self.pending_updates_done = Event()
        self.pending_updates_done.set()

        with torch.no_grad():
            averaged_tensors = [tensor.detach().cpu().float().clone() for tensor in self.local_tensors()]

        self.weight_previous_outer_step = [tensor.detach().cpu().float().clone() for tensor in self.local_tensors()]
        
        self.averaged_weights_for_inference = averaged_tensors
        
        super().__init__(averaged_tensors=averaged_tensors, tensor_infos=list(self.tensor_infos()), **kwargs)
        
        # Log the actual compression being used
        if hasattr(self, '_compression') and self._compression is not None:
            logger.info(f"[COMPRESSION] Averager using compression: {type(self._compression).__name__}")
        else:
            logger.info(f"[COMPRESSION] Averager compression: {kwargs.get('compression', 'not specified')}")

    def step(self, num_peers: int, data_lock: Optional[Lock] = None, wait: bool = True, **kwargs):
        """
        Average optimizer weights and gradients with peers.

        :param data_lock: averager locks it when model parameters are modified. Otherwise it's assumed that no model
        modifications occur during averaging step
        """
        if not wait:
            return self.step_executor.submit(self.step, data_lock, wait=True, **kwargs)

        # if data_lock is supplied, tensors might change during averaging, so we need to copy them
        if data_lock is None:
            data_lock = nullcontext()

        local_tensors = list(self.local_tensors())
        
        # Profiling: compute tensor sizes before averaging
        num_tensors = len(local_tensors)
        raw_bytes = sum(t.numel() * t.element_size() for t in local_tensors)
        # Estimate compressed size (FP16 = 2 bytes, 8-bit = 1 byte per element)
        num_elements = sum(t.numel() for t in local_tensors)
        fp16_bytes = num_elements * 2
        int8_bytes = num_elements * 1
        logger.info(
            f"[ALLREDUCE] Preparing to average {num_tensors} tensors: "
            f"raw={raw_bytes / 1e6:.2f}MB, "
            f"fp16={fp16_bytes / 1e6:.2f}MB, "
            f"int8={int8_bytes / 1e6:.2f}MB, "
            f"num_peers={num_peers}"
        )
        
        t_allreduce_start = time.perf_counter()
        
        with self.lock_averager_step, torch.no_grad():
            # Get previous averaged tensors and fill averager's tensors with current local tensors
            self.pending_updates_done.clear()
            with data_lock, self.get_tensors() as averaged_tensors:
                assert len(local_tensors) == len(averaged_tensors), (
                    "The number of optimized parameters should not change."
                )
                for averaged_tensor, local_tensor in zip(averaged_tensors, local_tensors):
                    averaged_tensor[...] = local_tensor.cpu().float()
            self.pending_updates_done.set()

            # I couldn't make `DecentralizedAverager` work in case there is only
            # one peer, so we handle it manually.
            if num_peers > 1:
                # Find a group and hopefully average tensors with peers, use batch sizes as weights
                gathered = super().step(**kwargs)
                t_allreduce = time.perf_counter() - t_allreduce_start
                
                # Profiling: log allreduce completion and effective bandwidth
                if gathered is not None:
                    # Each peer sends/receives approximately (n-1)/n of the data in ring allreduce
                    # But for simplicity, estimate as sending full data to all peers
                    effective_bandwidth_mbps = (fp16_bytes / 1e6) / t_allreduce if t_allreduce > 0 else float("inf")
                    logger.info(
                        f"[ALLREDUCE] Completed in {t_allreduce:.2f}s with {len(gathered)} peers, "
                        f"effective={effective_bandwidth_mbps:.2f} MB/s"
                    )
                else:
                    logger.warning(
                        f"[ALLREDUCE] Failed after {t_allreduce:.2f}s (gathered=None)"
                    )
            else:
                # Right now we only use the length of the list `gathered` in the code below, so this
                # works:
                gathered = ["DummyValue"]
                logger.info(f"[ALLREDUCE] Skipped (only 1 peer), no data exchange needed")
        
            if gathered is not None:
                # load averaged tensors back into model
                self.pending_updates_done.clear()
                with data_lock, self.get_tensors() as averaged_tensors:
                    if len(averaged_tensors) != len(local_tensors):
                        raise RuntimeError("The number of optimized parameters should not change")

                    self.averaged_weights_for_inference = [t.detach().cpu().float().clone() for t in averaged_tensors]
                    
                    # Set up the weights and their gradients for the optimizer step
                    for averaged_tensor, local_tensor, weight_previous_outer_step in zip(
                        averaged_tensors, local_tensors, self.weight_previous_outer_step
                    ):
                        if any(local_tensor is a for a in self.avg_only_params_list):
                            local_tensor[...] = averaged_tensor
                            local_tensor.grad = torch.zeros_like(local_tensor).to(**kw)
                        else:
                            kw = {
                                "device": local_tensor.device,
                                "dtype": local_tensor.dtype,
                            }
                            delta = weight_previous_outer_step.to(**kw) - averaged_tensor.to(**kw)
                            
                            local_tensor[...] = weight_previous_outer_step
                            local_tensor.grad = delta

                    
                    # Now apply the optimizer step with the pseudo-gradients
                    self.opt.step()
                    self.opt.zero_grad()

                    self.weight_previous_outer_step = [tensor.detach().cpu().float().clone() for tensor in self.local_tensors()]

                    self.pending_updates_done.set()

            self.outer_step += 1
            return gathered

    def local_tensors(self) -> Iterator[torch.Tensor]:
        """Iterate local trainer's tensors that should be averaged with peers"""
        for param_group in self.opt.param_groups:
            yield from param_group["params"]
        for stats in self.opt_statistics:
            for param_group in self.opt.param_groups:
                for param in param_group["params"]:
                    yield self.opt.state[param][stats]
        yield from iter(self.extra_tensors)

    def get_averaged_model_weights(self) -> list:
        """
        Return the globally averaged model weights from the last outer step.
        
        Returns:
            List of tensors containing the averaged weights (CPU, float32).
            Returns None if no outer step has been performed yet.
        """
        if not hasattr(self, 'averaged_weights_for_inference') or self.averaged_weights_for_inference is None:
            return None
        return self.averaged_weights_for_inference

    @contextmanager
    def use_averaged_weights_for_inference(self):
        """
        Context manager that temporarily loads averaged model weights for inference.
        
        This is useful for distributed RL where you want rollouts collected under
        the globally averaged policy (for on-policy training) while keeping the
        local divergent weights for gradient computation.
        
        Usage:
            with averager.use_averaged_weights_for_inference():
                # Model now has averaged weights (identical across all servers)
                actions = model(observations)
            # Model weights restored to local training state
        
        Note: This should NOT be used during training (backward pass) as it would
        corrupt the local model state needed for DiLoCo's pseudo-gradient computation.
        """
        if not hasattr(self, 'averaged_weights_for_inference') or self.averaged_weights_for_inference is None:
            # No averaging done yet, just yield without changes
            yield
            return
        
        local_tensors = list(self.local_tensors())
        
        # Save current local weights
        saved_local_weights = [t.detach().clone() for t in local_tensors]
        
        try:
            # Load averaged weights into model
            with torch.no_grad():
                for local_tensor, avg_tensor in zip(local_tensors, self.averaged_weights_for_inference):
                    local_tensor.copy_(avg_tensor.to(local_tensor.device, local_tensor.dtype))
            
            yield
        finally:
            # Restore local training weights
            with torch.no_grad():
                for local_tensor, saved_tensor in zip(local_tensors, saved_local_weights):
                    local_tensor.copy_(saved_tensor)
            
    def tensor_infos(self):
        """Get CompressionInfo for each tensor, accounting for its role and specification"""
        params = tuple(param for param_group in self.opt.param_groups for param in param_group["params"])
        assert len(params) == len(self.parameter_names)
        for param, key in zip(params, self.parameter_names):
            yield CompressionInfo.from_tensor(param, key=key, role=TensorRole.PARAMETER)
        for stats in self.opt_statistics:
            for param, key in zip(params, self.parameter_names):
                yield CompressionInfo.from_tensor(
                    self.opt.state[param][stats], key=(key, stats), role=TensorRole.OPTIMIZER
                )
        for i, extra_tensor in enumerate(self.extra_tensors):
            yield CompressionInfo.from_tensor(extra_tensor, key=i, role=TensorRole.UNSPECIFIED)

    def get_current_state(self):
        """
        Get current model/optimizer state and when requested by a newbie peer. executed in the host process.
        :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
        """
        t_start = time.perf_counter()
        with torch.no_grad():
            t_copy_start = time.perf_counter()
            optimized_parameters = tuple(
                param.detach().cpu() for param_group in self.opt.param_groups for param in param_group["params"]
            )
            t_copy_params = time.perf_counter() - t_copy_start

            parameter_infos = [
                CompressionInfo.from_tensor(param, key=key, role=TensorRole.PARAMETER)
                for param, key in zip(optimized_parameters, self.parameter_names)
            ]

            t_extra_start = time.perf_counter()
            extra_tensors = tuple(tensor.detach().cpu() for tensor in self.extra_tensors)
            t_copy_extra = time.perf_counter() - t_extra_start

            extra_infos = [
                CompressionInfo.from_tensor(extra_tensor, key=i, role=TensorRole.UNSPECIFIED)
                for i, extra_tensor in enumerate(extra_tensors)
            ]

            t_opt_start = time.perf_counter()
            optimizer_metadata, optimizer_tensors = dump_optimizer_state(self.opt)
            t_dump_opt = time.perf_counter() - t_opt_start

            optimizer_infos = [
                CompressionInfo.from_tensor(opt_tensor, key=i, role=TensorRole.OPTIMIZER)
                for i, opt_tensor in enumerate(optimizer_tensors)
            ]

        metadata = dict(step=self.outer_step, group_bits=self.get_group_bits(), optimizer_metadata=optimizer_metadata)
        all_tensors = list(chain(optimized_parameters, extra_tensors, optimizer_tensors))
        all_tensor_infos = list(chain(parameter_infos, extra_infos, optimizer_infos))

        # Profiling: compute sizes and log
        param_bytes = sum(p.numel() * p.element_size() for p in optimized_parameters)
        extra_bytes = sum(t.numel() * t.element_size() for t in extra_tensors)
        opt_bytes = sum(t.numel() * t.element_size() for t in optimizer_tensors)
        total_bytes = param_bytes + extra_bytes + opt_bytes
        t_total = time.perf_counter() - t_start

        logger.info(
            f"[UPLOAD] get_current_state: "
            f"params={param_bytes / 1e6:.2f}MB (copy={t_copy_params:.3f}s), "
            f"extra={extra_bytes / 1e6:.2f}MB (copy={t_copy_extra:.3f}s), "
            f"optim={opt_bytes / 1e6:.2f}MB (dump={t_dump_opt:.3f}s), "
            f"total={total_bytes / 1e6:.2f}MB in {t_total:.3f}s"
        )

        return metadata, all_tensors, all_tensor_infos

    def load_state_from_peers(self, **kwargs):
        """
        Attempt to download the latest optimizer state from peers and update trainer parameters/statistics.
        :returns: whether or the averager succeeded in loading parameters
        """
        t_start = time.perf_counter()
        parameters_and_extras = [param for param_group in self.opt.param_groups for param in param_group["params"]]
        parameters_and_extras.extend(self.extra_tensors)
        num_local_tensors = len(parameters_and_extras)

        t_download_start = time.perf_counter()
        loaded_state = super().load_state_from_peers(**kwargs)
        t_download = time.perf_counter() - t_download_start

        if loaded_state is None:
            logger.warning(f"[DOWNLOAD] load_state_from_peers: no state received after {t_download:.3f}s")
            return

        metadata, flat_tensors = loaded_state
        loaded_parameters_and_extras = flat_tensors[:num_local_tensors]
        loaded_opt_tensors = flat_tensors[num_local_tensors:]

        # Profiling: compute received sizes
        param_bytes = sum(t.numel() * t.element_size() for t in loaded_parameters_and_extras)
        opt_bytes = sum(t.numel() * t.element_size() for t in loaded_opt_tensors)
        total_bytes = param_bytes + opt_bytes

        t_apply_start = time.perf_counter()
        with torch.no_grad():
            for local_param, loaded_param in zip(parameters_and_extras, loaded_parameters_and_extras):
                local_param[...] = loaded_param
            load_optimizer_state(self.opt, metadata["optimizer_metadata"], loaded_opt_tensors)
        t_apply = time.perf_counter() - t_apply_start

        self.outer_step = max(self.outer_step, metadata["step"])
        
        self.averaged_weights_for_inference = [tensor.detach().cpu().float().clone() for tensor in self.local_tensors()]

        t_total = time.perf_counter() - t_start
        bandwidth_mbps = (total_bytes / 1e6) / t_download if t_download > 0 else float("inf")

        logger.info(
            f"[DOWNLOAD] load_state_from_peers: "
            f"params={param_bytes / 1e6:.2f}MB, optim={opt_bytes / 1e6:.2f}MB, "
            f"total={total_bytes / 1e6:.2f}MB, "
            f"download={t_download:.3f}s ({bandwidth_mbps:.2f} MB/s), "
            f"apply={t_apply:.3f}s, total={t_total:.3f}s, "
            f"loaded outer_step={metadata['step']}"
        )


class DiLoCoOptimizer(DecentralizedOptimizerBase):
    """
    TODO
    """

    def __init__(
        self,
        inner_optimizer: Union[TorchOptimizer, OptimizerFactory],
        outer_optimizer: Union[TorchOptimizer, OptimizerFactory],
        *,
        params: Optional[Iterable[torch.Tensor]] = None,
        avg_only_params: Optional[Iterable[torch.Tensor]] = None,
        dht: DHT,
        run_id: str,
        num_inner_steps: int,
        batch_size_per_step: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        scheduler: Optional[Union[str, LRSchedulerBase]] = None,
        num_warmup_steps: int = 0,
        num_total_steps: int = 0,
        min_local_steps: Optional[int] = None,
        min_refresh_period: float = 0.5,
        max_refresh_period: float = 30,
        default_refresh_period: float = 3,
        expected_drift_peers: float = 3,
        expected_drift_rate: float = 0.2,
        performance_ema_alpha: float = 0.1,
        metadata_expiration: float = 60.0,
        averaging_timeout: Optional[float] = None,
        load_state_timeout: float = 600.0,
        step_tolerance: int = 1,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = False,
        verbose: bool = False,
        expert: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        # convert params to list to provide params to both optimizers
        params_list = list(params)
    
        if isinstance(inner_optimizer, Callable):
            inner_optimizer = inner_optimizer(params=params_list, expert=expert) if expert is not None else inner_optimizer(params=params_list)
        if isinstance(outer_optimizer, Callable):
            self.outer_optimizer = outer_optimizer(params=params_list)

        super().__init__(inner_optimizer, dht)

        signature_validator = RSASignatureValidator()
        self._local_public_key = signature_validator.local_public_key
        dht.add_validators([SchemaValidator(TrainingProgressSchema, prefix=run_id), signature_validator])

        if reuse_grad_buffers and accumulate_grads_on is not None:
            logger.warning("Setting 'accumulate_grads_on' has no effect if reuse_grad_buffers=True")
        self.run_id = run_id
        self.scheduler = self._init_scheduler(
            scheduler=scheduler,
            num_warmup_steps=num_warmup_steps,
            num_total_steps=num_total_steps,
        )
        self.num_inner_steps, self.batch_size_per_step = num_inner_steps, batch_size_per_step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.min_refresh_period, self.max_refresh_period, self.default_refresh_period = (
            min_refresh_period,
            max_refresh_period,
            default_refresh_period,
        )
        self.expected_drift_peers, self.expected_drift_rate = expected_drift_peers, expected_drift_rate
        self.averaging_timeout = averaging_timeout
        self.load_state_timeout = load_state_timeout
        self.metadata_expiration = metadata_expiration
        self._grads, self.reuse_grad_buffers, self.accumulate_grads_on = None, reuse_grad_buffers, accumulate_grads_on
        self.client_mode, self.step_tolerance = client_mode, step_tolerance
        self.status_loglevel = logging.INFO if verbose else logging.DEBUG
        self.averager = self._make_averager(avg_only_params=avg_only_params, **kwargs)

        self.training_progress_key = f"{self.run_id}_progress"
        self.inner_step = 0  # number of inner steps since last outer step
        self.steps_accumulated = 0
        self.performance_ema = PerformanceEMA(alpha=performance_ema_alpha)
        self.last_step_time = None
        self.min_local_steps = min_local_steps if min_local_steps is not None else self.num_inner_steps - 1

        self.collaboration_state = self._fetch_state()
        self.lock_collaboration_state, self.collaboration_state_updated = Lock(), Event()
        self.lock_local_progress, self.should_report_progress = Lock(), Event()
        self._shutdown_evt = Event()
        self.progress_reporter = Thread(target=self.report_training_progress, daemon=True, name=f"{self}.reporter")
        self.progress_reporter.start()
        self.collaboration_state_updater = Thread(
            target=self.check_collaboration_state_periodically, daemon=True, name=f"{self}.collaboration_state_updater"
        )
        self.collaboration_state_updater.start()

    def _make_averager(self, avg_only_params: Optional[Iterable[torch.Tensor]] = None, **kwargs):
        """Create and configure the TrainingAverager instance for parameter and gradient averaging
        
        :param kwargs: additional parameters forwarded to TrainingAverager
        :returns: configured TrainingAverager instance
        """
        return DiLoCoTrainingAverager(
            optimizer=self.outer_optimizer,
            dht=self.dht,
            prefix=f"{self.run_id}_averaging",
            allreduce_timeout=self.averaging_timeout,
            client_mode=self.client_mode,
            avg_only_params=avg_only_params,
            **kwargs,
        )

    @property
    def outer_step(self) -> int:
        return self.averager.outer_step

    @property
    def is_synchronized(self) -> bool:
        return self.outer_step >= self.collaboration_state.outer_step - self.step_tolerance

    def is_alive(self) -> bool:
        return self.averager.is_alive()

    def load_state_from_peers(self, **kwargs):
        """Attempt to fetch the newest collaboration state from other peers"""
        with self.lock_collaboration_state:
            while True:
                try:
                    self.averager.load_state_from_peers(timeout=self.load_state_timeout, **kwargs)
                    break
                except BaseException as e:
                    logger.exception(f"Failed to load state from peers: {e}, retrying ...")
                    continue

            self.steps_accumulated = 0
            self.reset_accumulated_grads_()
            self.update_scheduler()

    def step(self, batch_size: Optional[int] = None, **kwargs):
        """
        Report accumulating gradients w.r.t. batch_size additional samples, optionally update model parameters

        :param batch_size: optional override for batch_size_per_step from init
        :param kwargs: additional parameters forwarded to TrainingAverager.step
        :returns: group_info dict containing information about the averaging group if a step was performed, None otherwise
        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """
        if self.batch_size_per_step is None:
            if batch_size is None:
                raise ValueError("Please either set batch_size_per_step parameter at init or when calling .step")
            logger.log(self.status_loglevel, f"Setting default batch_size_per_step to {batch_size}")
            self.batch_size_per_step = batch_size
        batch_size = batch_size if batch_size is not None else self.batch_size_per_step
        self.steps_accumulated += 1
        if self.steps_accumulated % self.gradient_accumulation_steps != 0:
            return

        if not self.is_synchronized:
            logger.log(self.status_loglevel, "Peer is out of sync.")
            self.load_state_from_peers()
            return

        if self.last_step_time is not None and get_dht_time() - self.last_step_time > self.metadata_expiration:
            logger.warning(
                f"Training step took {get_dht_time() - self.last_step_time}, "
                f"but metadata expired in {self.metadata_expiration} s."
            )

        self.step_inner(**kwargs)

        if self.collaboration_state.ready_for_outer_step(self.inner_step, min_local_steps=self.min_local_steps):
            self.step_outer(**kwargs)

    def step_inner(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        with self.lock_local_progress:
            self.inner_step += 1
            self.performance_ema.update(task_size=self.batch_size_per_step)
            self.should_report_progress.set()

        self.update_scheduler()
        if self.scheduler is not None:
            logger.info(f"Scheduler: {self.scheduler.get_last_lr()}, step: {self.scheduler._step_count}")

    def step_outer(self, **kwargs):
        logger.info(f"Running outer step after {self.inner_step + 1} inner steps")

        logger.log(self.status_loglevel, f"{self.run_id} beginning global step #{self.collaboration_state.outer_step}")
        self.collaboration_state = self._fetch_state()
        self.collaboration_state_updated.set()

        if not self.is_synchronized:
            self.load_state_from_peers()
            return

        with self.performance_ema.pause(), self.lock_collaboration_state:
            group_info = None

            num_peers = self.collaboration_state.num_peers
            if num_peers > 0:
                if self.inner_step <= 0:
                    logger.log(self.status_loglevel, "Skipped averaging: no inner steps")
                    return
                if self.collaboration_state.inner_steps_accumulated <= 0:
                    logger.log(self.status_loglevel, "Skipped averaging: no accumulated steps")
                    return
                mean_steps_per_worker = self.collaboration_state.inner_steps_accumulated / num_peers
                weight = (self.inner_step + 1) / mean_steps_per_worker
                logger.info(f"weight: {weight:.3f}")
                logger.info(f"batch_size_per_step: {self.batch_size_per_step}")
                
                logger.info(
                    f"[AVERAGING] {self.run_id} entering matchmaking: "
                    f"num_peers={num_peers}, weight={weight:.3f}, timeout={self.averaging_timeout}s"
                )
                t_avg_start = time.perf_counter()
                
                try:
                    group_info = self.averager.step(weight=weight, timeout=self.averaging_timeout, num_peers=num_peers, **kwargs)
                    t_avg = time.perf_counter() - t_avg_start
                    if group_info:
                        logger.info(
                            f"[AVERAGING] {self.run_id} averaged successfully with {len(group_info)} peers "
                            f"in {t_avg:.2f}s"
                        )
                    else:
                        logger.warning(
                            f"[AVERAGING] {self.run_id} averager.step returned None after {t_avg:.2f}s "
                            f"(no group formed?)"
                        )
                except BaseException as e:
                    t_avg = time.perf_counter() - t_avg_start
                    logger.warning(
                        f"[AVERAGING] {self.run_id} averaging failed after {t_avg:.2f}s: {type(e).__name__}: {e}"
                    )

                logger.log(self.status_loglevel, f"Outer optimizer step: done!")
                self.collaboration_state.register_step(self.outer_step + 1)
                self.collaboration_state_updated.set()

        self.inner_step = 0

        return group_info

    def _grad_buffers(self) -> Iterator[torch.Tensor]:
        """pytorch-internal gradient buffers"""
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    yield torch.zeros_like(param)
                else:
                    yield param.grad

    @torch.no_grad()
    def accumulated_grads(self) -> Iterator[torch.Tensor]:
        """local gradient accumulators"""
        if self.reuse_grad_buffers:
            yield from self._grad_buffers()
        elif self._grads is None:
            with torch.no_grad():
                self._grads = [
                    torch.zeros_like(grad, device=self.accumulate_grads_on) for grad in self._grad_buffers()
                ]
        yield from self._grads

    @torch.no_grad()
    def accumulate_grads_(self, batch_size: int):
        """add current gradients to grad accumulators (if any)"""
        if self.reuse_grad_buffers:
            return  # user is responsible for accumulating gradients in .grad buffers
        alpha = float(batch_size) / self.batch_size_per_step
        for grad_buf, grad_acc in zip(self._grad_buffers(), self.accumulated_grads()):
            grad_acc.add_(grad_buf.to(grad_acc.device), alpha=alpha)

    @torch.no_grad()
    def apply_accumulated_grads_(self, scale_by: Optional[float] = None):
        """Apply accumulated gradients to parameter .grad buffers, optionally scaling them
        
        :param scale_by: optional scaling factor to multiply gradients by
        """
        if self.reuse_grad_buffers:
            return
        for grad_buf, grad_acc in zip(self._grad_buffers(), self.accumulated_grads()):
            grad_buf[...] = grad_acc.to(grad_buf.device)
            if scale_by is not None:
                grad_buf.mul_(scale_by)

    @torch.no_grad()
    def reset_accumulated_grads_(self):
        """Reset all accumulated gradients to zero"""
        if self.reuse_grad_buffers:
            self.optimizer.zero_grad()
        else:
            for grad_buf in self.accumulated_grads():
                grad_buf.zero_()

    def report_training_progress(self):
        """Periodically publish metadata and the current number of samples accumulated towards the next step"""
        logger.info(f"[PROGRESS] Reporter thread started for {self.run_id}")
        backoff_s = 1.0
        max_backoff_s = 30.0
        last_report_time = None
        while self.is_alive() and not self._shutdown_evt.is_set():
            # Wait until progress should be reported, but also allow clean shutdown.
            self.should_report_progress.wait(timeout=1.0)
            if self._shutdown_evt.is_set() or not self.is_alive():
                break
            if not self.should_report_progress.is_set():
                # Log if we haven't reported in a while (possible stuck training loop)
                if last_report_time is not None:
                    time_since_last = get_dht_time() - last_report_time
                    if time_since_last > 60:
                        logger.warning(
                            f"[PROGRESS] {self.run_id} hasn't reported progress in {time_since_last:.0f}s "
                            f"(training loop may be blocked)"
                        )
                continue
            self.should_report_progress.clear()
            last_report_time = get_dht_time()
            with self.lock_local_progress:
                current_time = get_dht_time()
                local_state_info = TrainingState(
                    peer_id=self.averager.peer_id.to_bytes(),
                    outer_step=self.outer_step,
                    inner_step=self.inner_step,
                    inner_steps_per_second=self.performance_ema.samples_per_second / self.batch_size_per_step,
                    time=current_time,
                    client_mode=self.averager.client_mode,
                )

            try:
                self.dht.store(
                    key=self.training_progress_key,
                    subkey=self._local_public_key,
                    value=local_state_info.dict(),
                    expiration_time=current_time + self.metadata_expiration,
                    return_future=True,
                )
                # Reset backoff after a successful store.
                backoff_s = 1.0
                logger.debug(
                    f"[PROGRESS] {self.run_id} reported: outer_step={local_state_info.outer_step}, "
                    f"inner_step={local_state_info.inner_step}, "
                    f"steps/s={local_state_info.inner_steps_per_second:.4f}, "
                    f"expires_in={self.metadata_expiration:.0f}s"
                )
            except OSError as e:
                # IMPORTANT: if we silently stop reporting, other peers will drop us after metadata_expiration.
                logger.warning(
                    "Failed to publish training progress to DHT (will retry): %r", e, exc_info=True
                )
            except Exception as e:
                # Catch any other exceptions that might kill the reporter thread silently
                logger.exception(
                    f"[PROGRESS] Unexpected error publishing training progress: {type(e).__name__}: {e}"
                )
                # Exponential backoff with jitter, but keep the reporter thread alive.
                sleep_s = min(max_backoff_s, backoff_s) * (0.8 + 0.4 * random.random())
                backoff_s = min(max_backoff_s, backoff_s * 2.0)
                # Allow shutdown to interrupt the sleep.
                self._shutdown_evt.wait(timeout=sleep_s)
                continue

    def check_collaboration_state_periodically(self):
        """
        Periodically check the training progress from all peers. Trigger update after num_inner_steps total steps
        """
        backoff_s = 1.0
        max_backoff_s = 30.0
        while self.is_alive() and not self._shutdown_evt.is_set():
            time_to_next_update = max(0.0, self.collaboration_state.next_fetch_time - get_dht_time())
            if self.collaboration_state_updated.wait(time_to_next_update):
                self.collaboration_state_updated.clear()
                continue  # if state was updated externally, reset timer

            with self.lock_collaboration_state:
                if self._shutdown_evt.is_set() or not self.is_alive():
                    break
                try:
                    self.collaboration_state = self._fetch_state()
                    backoff_s = 1.0
                except OSError as e:
                    logger.warning(
                        "Failed to refresh collaboration state from DHT (will retry): %r", e, exc_info=True
                    )
                    sleep_s = min(max_backoff_s, backoff_s) * (0.8 + 0.4 * random.random())
                    backoff_s = min(max_backoff_s, backoff_s * 2.0)
                    # Let shutdown interrupt the backoff wait.
                    self._shutdown_evt.wait(timeout=sleep_s)
                    continue

    def _fetch_state(self) -> CollaborationState:
        """Read performance statistics reported by peers, estimate progress towards next batch
        
        :returns: CollaborationState containing current collaboration statistics and timing information
        """
        response, _expiration = self.dht.get(self.training_progress_key, latest=True) or (None, -float("inf"))
        current_time = get_dht_time()

        if not isinstance(response, dict) or not response:
            logger.log(
                self.status_loglevel,
                f"{self.run_id} found no active peers {f': {response}' if response else ''}",
            )
            local_eta_next_step = (
                max(0, self.num_inner_steps - self.inner_step) / self.performance_ema.samples_per_second / self.batch_size_per_step
            )
            return CollaborationState(
                outer_step=self.outer_step,
                inner_steps_accumulated=self.inner_step,
                num_inner_steps=self.num_inner_steps,
                num_peers=0,
                num_clients=0,
                eta_next_step=current_time + local_eta_next_step,
                next_fetch_time=current_time + self.default_refresh_period,
                any_peer_ready=False,
            )

        valid_peer_states = []
        for subkey, peer_state in response.items():
            if peer_state.value is None:
                logger.debug(f"[PEERS] Skipping peer subkey={subkey[:16]}...: value is None")
                continue
            try:
                state = TrainingState.parse_obj(peer_state.value)
                age_s = current_time - state.time
                # Log each peer's state for debugging
                logger.debug(
                    f"[PEERS] Peer {state.peer_id[:16].hex()}...: "
                    f"outer_step={state.outer_step}, inner_step={state.inner_step}, "
                    f"steps/s={state.inner_steps_per_second:.4f}, age={age_s:.1f}s, "
                    f"client_mode={state.client_mode}"
                )
                # Warn if peer state is very stale (> 5 min old)
                if age_s > 300:
                    logger.warning(
                        f"[PEERS] Stale peer {state.peer_id[:16].hex()}...: "
                        f"last reported {age_s:.0f}s ago (will expire after {self.metadata_expiration:.0f}s)"
                    )
                valid_peer_states.append(state)
            except Exception as e:
                logger.warning(f"[PEERS] Failed to parse peer state from subkey={subkey[:16]}...: {e}")

        num_peers = len(valid_peer_states)
        num_clients = sum(state.client_mode for state in valid_peer_states)
        global_outer_step = self.outer_step
        for state in valid_peer_states:
            if not state.client_mode:
                global_outer_step = max(global_outer_step, state.outer_step)

        inner_steps_accumulated = estimated_current_steps = total_steps_per_second = 0
        any_peer_ready = False  # Track if any peer has completed their inner steps

        for state in valid_peer_states:
            total_steps_per_second += state.inner_steps_per_second
            if state.outer_step == global_outer_step:
                inner_steps_accumulated += state.inner_step + 1
                estimated_current_steps += (
                    state.inner_step + 1 + max(0, current_time - state.time) * state.inner_steps_per_second
                )
                # Check if this peer has completed their inner steps (ready to trigger averaging)
                if state.inner_step + 1 >= self.num_inner_steps:
                    any_peer_ready = True
                    logger.debug(
                        f"[PEERS] Peer {state.peer_id[:16].hex()}... is ready for averaging "
                        f"(inner_step={state.inner_step}, target={self.num_inner_steps})"
                    )
            # note: we deliberately count only valid peers for samples_accumulated, but all peers for performance;
            # the rationale behind this is that outdated peers will synchronize and begin contributing shortly.

        estimated_steps_remaining = num_peers * self.num_inner_steps - inner_steps_accumulated
        estimated_time_to_next_step = max(0, estimated_steps_remaining) / max(total_steps_per_second, 1e-32)

        expected_max_peers = max(num_peers + self.expected_drift_peers, num_peers * (1 + self.expected_drift_rate))
        time_to_next_fetch = float(
            np.clip(
                a=estimated_time_to_next_step * num_peers / expected_max_peers,
                a_min=self.min_refresh_period,
                a_max=self.max_refresh_period,
            )
        )
        logger.log(
            self.status_loglevel,
            f"[inner_step {self.inner_step}] {self.run_id} accumulated {inner_steps_accumulated} steps from "
            f"{num_peers} peers for step #{global_outer_step}. "
            f"ETA {estimated_time_to_next_step:.2f} sec (refresh in {time_to_next_fetch:.2f} sec)"
            f"{' [LEADER READY]' if any_peer_ready else ''}",
        )
        return CollaborationState(
            outer_step=global_outer_step,
            inner_steps_accumulated=inner_steps_accumulated,
            num_inner_steps=self.num_inner_steps,
            num_peers=num_peers,
            num_clients=num_clients,
            eta_next_step=current_time + estimated_time_to_next_step,
            next_fetch_time=current_time + time_to_next_fetch,
            any_peer_ready=any_peer_ready,
        )

    def zero_grad(self, *args, **kwargs):
        if self.reuse_grad_buffers:
            raise ValueError(
                f"When running {self.__class__.__name__} with reuse_grad_buffers=True, user should never "
                f"call zero_grad manually. Gradients will be refreshed internally."
            )
        return self.optimizer.zero_grad(*args, **kwargs)

    def get_averaged_model_weights(self):
        """
        Get the globally averaged model weights from the last outer step.
        
        Useful for distributed RL where you want to generate rollouts under
        a consistent policy across all workers.
        
        Returns:
            List of averaged weight tensors, or None if no outer step performed yet.
        """
        return self.averager.get_averaged_model_weights()

    def use_averaged_weights_for_inference(self):
        """
        Context manager to temporarily use averaged weights for inference.
        
        This enables on-policy data collection in distributed RL by using
        the globally synchronized model for rollouts while preserving
        local weights for training gradients.
        
        Example:
            with optimizer.use_averaged_weights_for_inference():
                rollout_data = collect_rollouts(model)  # Uses averaged weights
            # Back to local training weights
            loss.backward()  # Gradients computed on local model
        """
        return self.averager.use_averaged_weights_for_inference()

    def update_scheduler(self):
        """Update the learning rate scheduler to match the current local step"""
        if self.scheduler:
            target_scheduler_step = self.outer_step * self.num_inner_steps + self.inner_step
            while self.scheduler._step_count < target_scheduler_step:
                self.scheduler.step()

    def _init_scheduler(
        self,
        *,
        scheduler: Optional[Union[str, LRSchedulerBase]],
        num_warmup_steps: int,
        num_total_steps: int,
    ):
        """
        Build a scheduler instance (or None).

        We intentionally schedule by OUTER steps (global updates): the scheduler is stepped until
        its internal step counter matches self.outer_step.
        """
        if scheduler is None:
            return None
        if isinstance(scheduler, str):
            if scheduler == "none":
                return None
            from distqat.scheduler import schedule_name_to_scheduler

            scheduler_factory = schedule_name_to_scheduler.get(scheduler)
            if scheduler_factory is None:
                raise ValueError(
                    f"Unknown scheduler {scheduler!r}. "
                    f"Valid: {sorted(k for k, v in schedule_name_to_scheduler.items() if v is not None)} + 'none'"
                )
            return scheduler_factory(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_total_steps,
            )
        # Assume caller passed an already-instantiated scheduler.
        return scheduler

    def shutdown(self):
        """Shutdown the collaborative optimizer, cleaning up resources and notifying peers"""
        # Stop background threads first so they don't touch DHT while we are shutting it down.
        self._shutdown_evt.set()
        # Wake any waiters so join doesn't stall.
        self.should_report_progress.set()
        self.collaboration_state_updated.set()
        logger.debug("Shutting down averager...")
        try:
            self.averager.shutdown()
        except Exception:
            pass
        logger.debug("Sending goodbye to peers...")
        try:
            self.dht.store(
                self.training_progress_key,
                subkey=self._local_public_key,
                value=None,
                expiration_time=get_dht_time() + self.metadata_expiration,
            )
        except OSError:
            pass
        self.collaboration_state_updater.join(timeout=10)
        self.progress_reporter.join(timeout=10)
        logger.debug(f"{self.__class__.__name__} is shut down.")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
