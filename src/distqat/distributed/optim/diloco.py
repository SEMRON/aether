import logging
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable, Dict, Iterable, Iterator, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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

    def ready_for_outer_step(self, local_inner_step: int):
        # Debug version: run outer step at exact number of inner steps
        return local_inner_step + 1 >= self.num_inner_steps
        # return self.num_peers > 0 and (
        #     self.inner_steps_accumulated >= self.num_peers * self.num_inner_steps
        #     or get_dht_time() >= self.eta_next_step
        # )

    def register_step(self, outer_step: int):
        self.outer_step = max(outer_step, self.outer_step)
        self.inner_steps_accumulated = 0
        self.eta_next_step = float("inf")

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

        super().__init__(averaged_tensors=averaged_tensors, tensor_infos=list(self.tensor_infos()), **kwargs)

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
            else:
                # Right now we only use the length of the list `gathered` in the code below, so this
                # works:
                gathered = ["DummyValue"]
        
            if gathered is not None:
                # load averaged tensors back into model
                self.pending_updates_done.clear()
                with data_lock, self.get_tensors() as averaged_tensors:
                    if len(averaged_tensors) != len(local_tensors):
                        raise RuntimeError("The number of optimized parameters should not change")

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
        with torch.no_grad():
            optimized_parameters = tuple(
                param.detach().cpu() for param_group in self.opt.param_groups for param in param_group["params"]
            )
            parameter_infos = [
                CompressionInfo.from_tensor(param, key=key, role=TensorRole.PARAMETER)
                for param, key in zip(optimized_parameters, self.parameter_names)
            ]
            extra_tensors = tuple(tensor.detach().cpu() for tensor in self.extra_tensors)
            extra_infos = [
                CompressionInfo.from_tensor(extra_tensor, key=i, role=TensorRole.UNSPECIFIED)
                for i, extra_tensor in enumerate(extra_tensors)
            ]
            optimizer_metadata, optimizer_tensors = dump_optimizer_state(self.opt)
            optimizer_infos = [
                CompressionInfo.from_tensor(opt_tensor, key=i, role=TensorRole.OPTIMIZER)
                for i, opt_tensor in enumerate(optimizer_tensors)
            ]

        metadata = dict(step=self.outer_step, group_bits=self.get_group_bits(), optimizer_metadata=optimizer_metadata)
        all_tensors = list(chain(optimized_parameters, extra_tensors, optimizer_tensors))
        all_tensor_infos = list(chain(parameter_infos, extra_infos, optimizer_infos))
        return metadata, all_tensors, all_tensor_infos

    def load_state_from_peers(self, **kwargs):
        """
        Attempt to download the latest optimizer state from peers and update trainer parameters/statistics.
        :returns: whether or the averager succeeded in loading parameters
        """
        parameters_and_extras = [param for param_group in self.opt.param_groups for param in param_group["params"]]
        parameters_and_extras.extend(self.extra_tensors)
        num_local_tensors = len(parameters_and_extras)

        loaded_state = super().load_state_from_peers(**kwargs)
        if loaded_state is None:
            return
        metadata, flat_tensors = loaded_state
        loaded_parameters_and_extras = flat_tensors[:num_local_tensors]
        loaded_opt_tensors = flat_tensors[num_local_tensors:]

        with torch.no_grad():
            for local_param, loaded_param in zip(parameters_and_extras, loaded_parameters_and_extras):
                local_param[...] = loaded_param
            load_optimizer_state(self.opt, metadata["optimizer_metadata"], loaded_opt_tensors)

        self.outer_step = max(self.outer_step, metadata["step"])


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
        scheduler: Optional[LRSchedulerBase] = None,
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
        **kwargs,
    ):
        # convert params to list to provide params to both optimizers
        params_list = list(params)
    
        if isinstance(inner_optimizer, Callable):
            inner_optimizer = inner_optimizer(params=params_list)
        if isinstance(outer_optimizer, Callable):
            self.outer_optimizer = outer_optimizer(params=params_list)

        super().__init__(inner_optimizer, dht)

        signature_validator = RSASignatureValidator()
        self._local_public_key = signature_validator.local_public_key
        dht.add_validators([SchemaValidator(TrainingProgressSchema, prefix=run_id), signature_validator])

        if reuse_grad_buffers and accumulate_grads_on is not None:
            logger.warning("Setting 'accumulate_grads_on' has no effect if reuse_grad_buffers=True")
        self.run_id, self.scheduler = run_id, scheduler
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

        self.collaboration_state = self._fetch_state()
        self.lock_collaboration_state, self.collaboration_state_updated = Lock(), Event()
        self.lock_local_progress, self.should_report_progress = Lock(), Event()
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

        if self.collaboration_state.ready_for_outer_step(self.inner_step):
            self.step_outer(**kwargs)

    def step_inner(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_scheduler()

        with self.lock_local_progress:
            self.inner_step += 1
            self.performance_ema.update(task_size=self.batch_size_per_step)
            self.should_report_progress.set()

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
                try:
                    group_info = self.averager.step(weight=weight, timeout=self.averaging_timeout, num_peers=num_peers, **kwargs)
                    if group_info:
                        logger.log(self.status_loglevel, f"Averaged tensors successfully with {len(group_info)} peers")
                except BaseException as e:
                    logger.log(self.status_loglevel, f"Skipped averaging: averaging round failed with {repr(e)}.")

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
        while self.is_alive():
            self.should_report_progress.wait()
            self.should_report_progress.clear()
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

            self.dht.store(
                key=self.training_progress_key,
                subkey=self._local_public_key,
                value=local_state_info.dict(),
                expiration_time=current_time + self.metadata_expiration,
                return_future=True,
            )

    def check_collaboration_state_periodically(self):
        """
        Periodically check the training progress from all peers. Trigger update after num_inner_steps total steps
        """
        while self.is_alive():
            time_to_next_update = max(0.0, self.collaboration_state.next_fetch_time - get_dht_time())
            if self.collaboration_state_updated.wait(time_to_next_update):
                self.collaboration_state_updated.clear()
                continue  # if state was updated externally, reset timer

            with self.lock_collaboration_state:
                self.collaboration_state = self._fetch_state()

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
            )

        valid_peer_states = [
            TrainingState.parse_obj(peer_state.value)
            for peer_state in response.values()
            if peer_state.value is not None
        ]

        num_peers = len(valid_peer_states)
        num_clients = sum(state.client_mode for state in valid_peer_states)
        global_outer_step = self.outer_step
        for state in valid_peer_states:
            if not state.client_mode:
                global_outer_step = max(global_outer_step, state.outer_step)

        inner_steps_accumulated = estimated_current_steps = total_steps_per_second = 0

        for state in valid_peer_states:
            total_steps_per_second += state.inner_steps_per_second
            if state.outer_step == global_outer_step:
                inner_steps_accumulated += state.inner_step + 1
                estimated_current_steps += (
                    state.inner_step + 1 + max(0, current_time - state.time) * state.inner_steps_per_second
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
            f"ETA {estimated_time_to_next_step:.2f} sec (refresh in {time_to_next_fetch:.2f} sec)",
        )
        return CollaborationState(
            outer_step=global_outer_step,
            inner_steps_accumulated=inner_steps_accumulated,
            num_inner_steps=self.num_inner_steps,
            num_peers=num_peers,
            num_clients=num_clients,
            eta_next_step=current_time + estimated_time_to_next_step,
            next_fetch_time=current_time + time_to_next_fetch,
        )

    def zero_grad(self, *args, **kwargs):
        if self.reuse_grad_buffers:
            raise ValueError(
                f"When running {self.__class__.__name__} with reuse_grad_buffers=True, user should never "
                f"call zero_grad manually. Gradients will be refreshed internally."
            )
        return self.optimizer.zero_grad(*args, **kwargs)

    def update_scheduler(self):
        """Update the learning rate scheduler to match the current local step"""
        if self.scheduler:
            while self.scheduler._step_count < self.outer_step:
                self.scheduler.step() 

    def shutdown(self):
        """Shutdown the collaborative optimizer, cleaning up resources and notifying peers"""
        logger.debug("Shutting down averager...")
        self.averager.shutdown()
        logger.debug("Sending goodbye to peers...")
        self.dht.store(
            self.training_progress_key,
            subkey=self._local_public_key,
            value=None,
            expiration_time=get_dht_time() + self.metadata_expiration,
        )
        self.collaboration_state_updater.join(timeout=5)
        self.progress_reporter.join(timeout=5)
        logger.debug(f"{self.__class__.__name__} is shut down.")

    def __del__(self):
        self.shutdown()
