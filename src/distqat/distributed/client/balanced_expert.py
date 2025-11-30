from typing import Any, Dict, Optional, Tuple

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

    @property
    def info(self):
        """
        Get expert metadata including input/output schemas from one of the available experts.
        
        :returns: Dictionary containing expert metadata (schemas, keyword names, etc.)
        """
        while self._expert_info is None:
            try:
                with self.expert_balancer.use_another_expert(1) as chosen_expert:
                    self._expert_info = chosen_expert.info
            except BaseException as e:
                logger.error(f"Tried to get expert info from {chosen_expert} but caught {repr(e)}")
        return self._expert_info


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
        logger.debug("TRAINER: Forward")
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        # detach to avoid pickling the computation graph
        ctx.expert_balancer, ctx.info = expert_balancer, info
        ctx.forward_timeout, ctx.backward_timeout = forward_timeout, backward_timeout
        ctx.forward_task_size, ctx.backward_task_size = forward_task_size, backward_task_size
        inputs = tuple(tensor.cpu().detach() for tensor in inputs)
        ctx.save_for_backward(*inputs)

        serialized_tensors = [
            serialize_torch_tensor(inp, proto.compression)
            for inp, proto in zip(inputs, nested_flatten(info["forward_schema"]))
        ]
        while True:
            try:
                with expert_balancer.use_another_expert(forward_task_size) as chosen_expert:
                    logger.debug(f"TRAINER: Forwarding to expert {chosen_expert.uid}")
                    forward_request = runtime_pb2.ExpertRequest(uid=chosen_expert.uid, tensors=serialized_tensors)
                    outputs = chosen_expert.stub.forward(forward_request, timeout=forward_timeout)
                break
            except KeyboardInterrupt:
                raise
            except BaseException:
                logger.exception(f"Tried to call forward for expert {chosen_expert}:")

        deserialized_outputs = [deserialize_torch_tensor(tensor) for tensor in outputs.tensors]
        return tuple(deserialized_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        logger.debug("TRAINER: Backward")

        grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
        inputs_and_grad_outputs = tuple(nested_flatten((ctx.saved_tensors, grad_outputs_cpu)))
        backward_schema = tuple(nested_flatten((ctx.info["forward_schema"], ctx.info["outputs_schema"])))
        serialized_tensors = [
            serialize_torch_tensor(tensor, proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        ]
        while True:
            try:
                with ctx.expert_balancer.use_another_expert(ctx.backward_task_size) as chosen_expert:
                    backward_request = runtime_pb2.ExpertRequest(uid=chosen_expert.uid, tensors=serialized_tensors)
                    grad_inputs = chosen_expert.stub.backward(backward_request, timeout=ctx.backward_timeout)
                break
            except KeyboardInterrupt:
                raise
            except BaseException:
                logger.exception(f"Tried to call backward for expert {chosen_expert}:")
        deserialized_grad_inputs = [deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors]
        return (DUMMY, None, None, None, None, None, None, *deserialized_grad_inputs)
