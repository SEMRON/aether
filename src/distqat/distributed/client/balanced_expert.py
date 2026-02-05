from typing import Any, Dict, Optional, Tuple
import os
import time

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.dht import DHT
from hivemind.proto import runtime_pb2
from hivemind.utils import get_logger
from hivemind.utils.nested import nested_compare, nested_flatten, nested_pack

from distqat.distributed.client.balancer import ExpertBalancer
from distqat.distributed.client.expert import DUMMY

logger = get_logger(__name__)
logger.setLevel("DEBUG")

# Enable extra timing breakdown logs for remote RPC calls.
# This is intentionally off by default because it adds some overhead and log noise.
_PROFILE_REMOTE_RPC = os.getenv("DISTQAT_PROFILE_REMOTE_RPC", "").lower() in ("1", "true", "yes", "on")

class BalancedRemoteExpert(nn.Module):
    """
    A torch module that dynamically assigns weights to one RemoteExpert from a pool, proportionally to their throughput.
    
    This module maintains a pool of remote experts and uses load balancing to select the most appropriate
    expert for each forward pass based on their current throughput and availability.
    
    :param dht: Distributed hash table for expert discovery and coordination
    :param uid_prefix: Prefix for expert UIDs to search for in the DHT
    :param forward_timeout: Maximum time to wait for forward pass completion, None for no timeout
    :param backward_timeout: Maximum time to wait for backward pass completion, None for no timeout
    :param update_period: How often to refresh the expert pool from DHT (in seconds)
    :param backward_task_size_multiplier: Multiplier for task size during backward pass load estimation
    :param kwargs: Additional keyword arguments passed to the ExpertBalancer
    """

    def __init__(
        self,
        *,
        dht: DHT,
        uid_prefix: str,
        forward_timeout: Optional[float] = None,
        backward_timeout: Optional[float] = None,
        update_period: float = 30.0,
        backward_task_size_multiplier: float = 2.5,
        **kwargs,
    ):
        super().__init__()

        self.dht, self.uid_prefix = dht, uid_prefix
        self.forward_timeout, self.backward_timeout = forward_timeout, backward_timeout
        self.backward_task_size_multiplier = backward_task_size_multiplier
        self.expert_balancer = ExpertBalancer(dht, key=self.uid_prefix, update_period=update_period, **kwargs)
        self._expert_info = None  # expert['info'] from one of experts in the grid

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        """
        Call one of the RemoteExperts for the specified inputs and return output. Compatible with pytorch.autograd.

        Selects the most appropriate expert from the pool based on load balancing and routes the computation
        to that expert. Only one expert is used per forward pass, not an average of multiple experts.

        :param args: input tensors that will be passed to the selected expert, batch-first
        :param kwargs: extra keyword tensors that will be passed to the selected expert, batch-first
        :returns: output from the selected expert, nested structure of batch-first tensors
        """
        assert len(kwargs) == len(self.info["keyword_names"]), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}

        if self._expert_info is None:
            raise NotImplementedError()
        # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors

        forward_inputs = (args, kwargs)

        if not nested_compare(forward_inputs, self.info["forward_schema"]):
            raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

        flat_inputs = list(nested_flatten(forward_inputs))
        forward_task_size = flat_inputs[0].shape[0]

        # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        flat_outputs = _BalancedRemoteModuleCall.apply(
            DUMMY,
            self.expert_balancer,
            self.info,
            self.forward_timeout,
            self.backward_timeout,
            forward_task_size,
            forward_task_size * self.backward_task_size_multiplier,
            *flat_inputs,
        )

        return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

    def forward_averaged(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        """
        Call one of the RemoteExperts using globally averaged model weights.
        
        This is an inference-only forward pass (no gradients) that uses the averaged
        model weights from the last DiLoCo outer step. Useful for distributed RL
        where rollouts should be collected under a consistent policy.

        :param args: input tensors that will be passed to the selected expert, batch-first
        :param kwargs: extra keyword tensors that will be passed to the selected expert, batch-first
        :returns: output from the selected expert, nested structure of batch-first tensors
        """
        assert len(kwargs) == len(self.info["keyword_names"]), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}

        if self._expert_info is None:
            raise NotImplementedError()

        forward_inputs = (args, kwargs)

        if not nested_compare(forward_inputs, self.info["forward_schema"]):
            raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

        flat_inputs = list(nested_flatten(forward_inputs))
        forward_task_size = flat_inputs[0].shape[0]

        # Call _forward_averaged_impl which uses the forward_averaged RPC endpoint
        flat_outputs = _forward_averaged_impl(
            self.expert_balancer,
            self.info,
            self.forward_timeout,
            forward_task_size,
            flat_inputs,
        )

        return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

    @property
    def info(self):
        """
        Get expert metadata including input/output schemas from one of the available experts.
        
        :returns: Dictionary containing expert metadata (schemas, keyword names, etc.)
        """
        while self._expert_info is None:
            chosen_expert = None
            try:
                with self.expert_balancer.use_another_expert(1) as chosen_expert:
                    self._expert_info = chosen_expert.info
            except BaseException as e:
                logger.error(f"Tried to get expert info from {chosen_expert} but caught {repr(e)}")
        return self._expert_info


def _forward_averaged_impl(
    expert_balancer,
    info,
    forward_timeout,
    forward_task_size,
    flat_inputs,
):
    """
    Implementation of forward_averaged that calls the forward_averaged RPC endpoint.
    This is inference-only (no gradients) so we don't need an autograd Function.
    """
    # Same transport downcast as regular forward: bf16 -> fp16
    if _PROFILE_REMOTE_RPC:
        t_copy_start = time.time()
    inputs = tuple(
        (
            tensor.cpu().detach().to(dtype=torch.float16)
            if tensor.dtype == torch.bfloat16
            else tensor.cpu().detach()
        )
        for tensor in flat_inputs
    )
    if _PROFILE_REMOTE_RPC:
        t_copy = time.time() - t_copy_start

    if _PROFILE_REMOTE_RPC:
        t_ser_start = time.time()
    serialized_tensors = [
        serialize_torch_tensor(inp, proto.compression)
        for inp, proto in zip(inputs, nested_flatten(info["forward_schema"]))
    ]
    if _PROFILE_REMOTE_RPC:
        t_ser = time.time() - t_ser_start
    
    input_bytes = sum(t.ByteSize() for t in serialized_tensors)
    
    while True:
        try:
            with expert_balancer.use_another_expert(forward_task_size) as chosen_expert:
                forward_request = runtime_pb2.ExpertRequest(uid=chosen_expert.uid, tensors=serialized_tensors)
                
                start_time = time.time()
                # Use forward_averaged RPC endpoint
                outputs = chosen_expert.stub.forward_averaged(forward_request, timeout=forward_timeout)
                elapsed = time.time() - start_time
                
                output_bytes = sum(t.ByteSize() for t in outputs.tensors)
                if _PROFILE_REMOTE_RPC:
                    logger.debug(
                        f"[TRAINER:ForwardAveraged] uid={chosen_expert.uid} "
                        f"sent={input_bytes / 1e6:.2f}MB recv={output_bytes / 1e6:.2f}MB "
                        f"rpc={elapsed:.4f}s cpu_copy={t_copy:.4f}s serialize={t_ser:.4f}s"
                    )
                else:
                    logger.debug(
                        f"[TRAINER:ForwardAveraged] uid={chosen_expert.uid} "
                        f"sent={input_bytes / 1e6:.2f}MB recv={output_bytes / 1e6:.2f}MB "
                        f"time={elapsed:.4f}s"
                    )
            break
        except KeyboardInterrupt:
            raise
        except BaseException:
            logger.exception(f"Tried to call forward_averaged for expert {chosen_expert}:")

    if _PROFILE_REMOTE_RPC:
        t_deser_start = time.time()
    deserialized_outputs = [deserialize_torch_tensor(tensor) for tensor in outputs.tensors]
    if _PROFILE_REMOTE_RPC:
        t_deser = time.time() - t_deser_start
        logger.debug(f"[TRAINER:ForwardAveraged] uid={chosen_expert.uid} deserialize={t_deser:.4f}s")
    return tuple(deserialized_outputs)


class _BalancedRemoteModuleCall(torch.autograd.Function):
    """Internal autograd-friendly call of a remote module. For applications, use BalancedRemoteExpert instead."""

    @staticmethod
    def forward(
        ctx,
        dummy: torch.Tensor,
        expert_balancer: ExpertBalancer,
        info: Dict[str, Any],
        forward_timeout: float,
        backward_timeout: float,
        forward_task_size: float,
        backward_task_size: float,
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        # detach to avoid pickling the computation graph
        ctx.expert_balancer, ctx.info = expert_balancer, info
        ctx.forward_timeout, ctx.backward_timeout = forward_timeout, backward_timeout
        ctx.forward_task_size, ctx.backward_task_size = forward_task_size, backward_task_size
        # Save original devices and dtypes to restore gradients in backward
        ctx.input_devices = [t.device for t in inputs]
        ctx.input_dtypes = [t.dtype for t in inputs]
        # NOTE: hivemind serialization does not support bf16 efficiently:
        # - with CompressionType.NONE, bf16 tensors are serialized as fp32-sized payloads (2x bigger than fp16)
        # - Float16Compression does not support bf16 tensors
        # To avoid doubling network traffic for bf16-mixed, we downcast bf16 -> fp16 for transport.
        if _PROFILE_REMOTE_RPC:
            t_copy_start = time.time()
        inputs = tuple(
            (
                tensor.cpu().detach().to(dtype=torch.float16)
                if tensor.dtype == torch.bfloat16
                else tensor.cpu().detach()
            )
            for tensor in inputs
        )
        if _PROFILE_REMOTE_RPC:
            t_copy = time.time() - t_copy_start
        ctx.save_for_backward(*inputs)

        if _PROFILE_REMOTE_RPC:
            t_ser_start = time.time()
        serialized_tensors = [
            serialize_torch_tensor(inp, proto.compression)
            for inp, proto in zip(inputs, nested_flatten(info["forward_schema"]))
        ]
        if _PROFILE_REMOTE_RPC:
            t_ser = time.time() - t_ser_start
        
        input_bytes = sum(t.ByteSize() for t in serialized_tensors)
        
        while True:
            try:
                with expert_balancer.use_another_expert(forward_task_size) as chosen_expert:
                    forward_request = runtime_pb2.ExpertRequest(uid=chosen_expert.uid, tensors=serialized_tensors)
                    
                    start_time = time.time()
                    outputs = chosen_expert.stub.forward(forward_request, timeout=forward_timeout)
                    elapsed = time.time() - start_time
                    
                    output_bytes = sum(t.ByteSize() for t in outputs.tensors)
                    if _PROFILE_REMOTE_RPC:
                        logger.debug(
                            f"[TRAINER:Forward] uid={chosen_expert.uid} "
                            f"sent={input_bytes / 1e6:.2f}MB recv={output_bytes / 1e6:.2f}MB "
                            f"rpc={elapsed:.4f}s cpu_copy={t_copy:.4f}s serialize={t_ser:.4f}s"
                        )
                    else:
                        logger.debug(
                            f"[TRAINER:Forward] uid={chosen_expert.uid} "
                            f"sent={input_bytes / 1e6:.2f}MB recv={output_bytes / 1e6:.2f}MB "
                            f"time={elapsed:.4f}s"
                        )
                break
            except KeyboardInterrupt:
                raise
            except BaseException:
                logger.exception(f"Tried to call forward for expert {chosen_expert}:")

        if _PROFILE_REMOTE_RPC:
            t_deser_start = time.time()
        deserialized_outputs = [deserialize_torch_tensor(tensor) for tensor in outputs.tensors]
        if _PROFILE_REMOTE_RPC:
            t_deser = time.time() - t_deser_start
            logger.debug(f"[TRAINER:Forward] uid={chosen_expert.uid} deserialize={t_deser:.4f}s")
        return tuple(deserialized_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:

        # Same transport downcast as in forward: keep wire payload small when upstream grads are bf16.
        if _PROFILE_REMOTE_RPC:
            t_copy_start = time.time()
        grad_outputs_cpu = tuple(
            (tensor.cpu().to(dtype=torch.float16) if tensor.dtype == torch.bfloat16 else tensor.cpu())
            for tensor in grad_outputs
        )
        if _PROFILE_REMOTE_RPC:
            t_copy = time.time() - t_copy_start
        inputs_and_grad_outputs = tuple(nested_flatten((ctx.saved_tensors, grad_outputs_cpu)))
        backward_schema = tuple(nested_flatten((ctx.info["forward_schema"], ctx.info["outputs_schema"])))
        if _PROFILE_REMOTE_RPC:
            t_ser_start = time.time()
        serialized_tensors = [
            serialize_torch_tensor(tensor, proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        ]
        if _PROFILE_REMOTE_RPC:
            t_ser = time.time() - t_ser_start
        
        input_bytes = sum(t.ByteSize() for t in serialized_tensors)
        
        while True:
            try:
                with ctx.expert_balancer.use_another_expert(ctx.backward_task_size) as chosen_expert:
                    backward_request = runtime_pb2.ExpertRequest(uid=chosen_expert.uid, tensors=serialized_tensors)
                    
                    start_time = time.time()
                    grad_inputs = chosen_expert.stub.backward(backward_request, timeout=ctx.backward_timeout)
                    elapsed = time.time() - start_time
                    
                    output_bytes = sum(t.ByteSize() for t in grad_inputs.tensors)
                    if _PROFILE_REMOTE_RPC:
                        logger.debug(
                            f"[TRAINER:Backward] uid={chosen_expert.uid} "
                            f"sent={input_bytes / 1e6:.2f}MB recv={output_bytes / 1e6:.2f}MB "
                            f"rpc={elapsed:.4f}s cpu_copy={t_copy:.4f}s serialize={t_ser:.4f}s"
                        )
                    else:
                        logger.debug(
                            f"[TRAINER:Backward] uid={chosen_expert.uid} "
                            f"sent={input_bytes / 1e6:.2f}MB recv={output_bytes / 1e6:.2f}MB "
                            f"time={elapsed:.4f}s"
                        )
                break
            except KeyboardInterrupt:
                raise
            except BaseException:
                logger.exception(f"Tried to call backward for expert {chosen_expert}:")
        if _PROFILE_REMOTE_RPC:
            t_deser_start = time.time()
        deserialized_grad_inputs = [deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors]
        if _PROFILE_REMOTE_RPC:
            t_deser = time.time() - t_deser_start
            logger.debug(f"[TRAINER:Backward] uid={chosen_expert.uid} deserialize={t_deser:.4f}s")
        # Move gradients back to original devices and dtypes to match forward inputs
        grad_inputs_restored = [
            grad.to(device=device, dtype=dtype)
            for grad, device, dtype in zip(deserialized_grad_inputs, ctx.input_devices, ctx.input_dtypes)
        ]
        return (DUMMY, None, None, None, None, None, None, *grad_inputs_restored)
