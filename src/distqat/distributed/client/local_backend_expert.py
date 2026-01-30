from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from hivemind.utils import get_logger
from hivemind.utils.nested import nested_compare, nested_flatten, nested_pack

from distqat.distributed.client.expert import DUMMY

logger = get_logger(__name__)
# logger.setLevel("DEBUG")

class LocalBackendExpert(nn.Module):
    """
    A drop-in replacement for BalancedRemoteExpert when the expert backend is available in-process.

    This bypasses gRPC/protobuf serialization entirely by submitting tasks directly into the expert's TaskPools.
    It preserves the same autograd contract as remote experts:
    - forward runs under no_grad inside ExpertBackend.forward
    - backward triggers ExpertBackend.backward (re-runs forward, computes input grads, applies optimizer step)
    """

    def __init__(self, *, backend: Any):
        super().__init__()
        self.backend = backend
        self._expert_info: Optional[Dict[str, Any]] = None

    @property
    def info(self) -> Dict[str, Any]:
        if self._expert_info is None:
            self._expert_info = self.backend.get_info()
        return self._expert_info

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        assert len(kwargs) == len(self.info["keyword_names"]), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}
        logger.debug(f"Forwarding inputs: {args}, {kwargs}")
        forward_inputs = (args, kwargs)
        if not nested_compare(forward_inputs, self.info["forward_schema"]):
            raise TypeError("Inputs do not match expert input schema. Did you pass the right number of parameters?")
        logger.debug(f"Forwarding inputs match schema")
        flat_inputs = list(nested_flatten(forward_inputs))
        logger.debug(f"Flat inputs: {flat_inputs}")
        flat_outputs = _LocalBackendModuleCall.apply(
            DUMMY,
            self.backend,
            self.info,
            *flat_inputs,
        )
        return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

    def forward_averaged(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        """
        Forward pass using globally averaged model weights.
        
        This is inference-only (no gradients) and uses the averaged weights from
        the last DiLoCo outer step if available.
        """
        assert len(kwargs) == len(self.info["keyword_names"]), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}
        forward_inputs = (args, kwargs)
        if not nested_compare(forward_inputs, self.info["forward_schema"]):
            raise TypeError("Inputs do not match expert input schema. Did you pass the right number of parameters?")
        flat_inputs = list(nested_flatten(forward_inputs))
        
        # Call ExpertBackend.forward_averaged directly (bypass TaskPool since we're in-process)
        # This avoids multiprocessing overhead and potential stalls (same pattern as forward/backward)
        inputs = tuple(t.detach() for t in flat_inputs)
        device = self.backend.device
        inputs_on_device = tuple(t.to(device) if t.device != device else t for t in inputs)
        flat_outputs = self.backend.forward_averaged(*inputs_on_device)
        return nested_pack(tuple(flat_outputs), structure=self.info["outputs_schema"])


class _LocalBackendModuleCall(torch.autograd.Function):
    """Internal autograd-friendly call of an in-process ExpertBackend."""

    @staticmethod
    def forward(
        ctx,
        dummy: torch.Tensor,
        backend: Any,
        info: Dict[str, Any],
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Detach to avoid building graphs across the expert boundary (matches remote semantics)
        ctx.backend, ctx.info = backend, info
        inputs = tuple(t.detach() for t in inputs)
        ctx.save_for_backward(*inputs)

        # Call ExpertBackend.forward directly (bypass TaskPool since we're in-process)
        # This avoids multiprocessing overhead and potential race conditions
        # Move inputs to the backend's device (mirrors what Runtime.load_batch_to_runtime does)
        device = backend.device
        inputs_on_device = tuple(t.to(device) if t.device != device else t for t in inputs)
        outputs = backend.forward(*inputs_on_device)
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        inputs = ctx.saved_tensors
        # submit inputs + grad_outputs (must match backend.backward_schema flattening)
        inputs_and_grad_outputs = tuple(nested_flatten((inputs, grad_outputs)))
        # Call ExpertBackend.backward directly (bypass TaskPool since we're in-process)
        # Move tensors to the backend's device (mirrors what Runtime.load_batch_to_runtime does)
        device = ctx.backend.device
        tensors_on_device = tuple(t.to(device) if t.device != device else t for t in inputs_and_grad_outputs)
        grad_inputs = ctx.backend.backward(*tensors_on_device)
        # Return grads for (dummy, backend, info, *inputs)
        return (DUMMY, None, None, *grad_inputs)

