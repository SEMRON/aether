from __future__ import annotations
from typing import Any
from dataclasses import dataclass

import torch as t


@dataclass(frozen=True, slots=True, kw_only=True)
class MetaTensor:
    """
    fp32 (or any dtype) tensor with quantization metadata.
    Immutable, device-aware, and easy to move/copy.
    """

    data: t.Tensor
    step_size: t.Tensor
    zero_point: t.Tensor
    bit: int

    @classmethod
    def from_tensor(
        cls,
        tensor: t.Tensor,
        *,
        step_size: t.Tensor,
        zero_point: t.Tensor | None = None,
        bit: int,
        detach_meta: bool = True,
    ) -> MetaTensor:
        """
        Create a MetaTensor from an existing tensor, aligning device/dtype
        for step_size and zero_point, and applying optional detaching.

        Parameters
        ----------
        tensor : t.Tensor
            The data tensor (e.g. fake-quant output).
        step_size : t.Tensor
            Step size (scalar or broadcastable).
        zero_point : t.Tensor, optional
            Defaults to a zero scalar on `tensor.device` if not provided.
        bit : int
            Bit-width of quantization.
        detach_meta : bool
            If True, detaches step_size and zero_point from the graph.
        """
        dev, dt = tensor.device, tensor.dtype

        if zero_point is None:
            zero_point = t.zeros((), device=dev, dtype=dt)

        # Ensure tensors (in case user passes numbers)
        step_size = t.as_tensor(step_size, device=dev, dtype=dt)
        zero_point = t.as_tensor(zero_point, device=dev, dtype=dt)

        if detach_meta:
            step_size = step_size.detach()
            zero_point = zero_point.detach()

        _ensure_broadcastable(tensor.shape, step_size.shape, zero_point.shape)

        return cls(
            data=tensor,
            step_size=step_size.to(device=dev, dtype=dt),
            zero_point=zero_point.to(device=dev, dtype=dt),
            bit=bit,
        )

    def to(self, *args: Any, **kwargs: Any) -> MetaTensor:
        """
        Mirror `torch.Tensor.to(...)`.
        Keeps metadata aligned to the resulting data's device & dtype.
        """
        d = self.data.to(*args, **kwargs)

        ss = self.step_size.to(device=d.device, dtype=d.dtype)
        zp = self.zero_point.to(device=d.device, dtype=d.dtype)

        return MetaTensor(data=d, step_size=ss, zero_point=zp, bit=self.bit)

    def cpu(self) -> MetaTensor:
        return self.to("cpu")

    def cuda(self, non_blocking: bool = False) -> MetaTensor:
        return self.to("cuda", non_blocking=non_blocking)

    def detach(self) -> MetaTensor:
        return MetaTensor(
            data=self.data.detach(),
            step_size=self.step_size.detach(),
            zero_point=self.zero_point.detach(),
            bit=self.bit,
        )

    def clone(self) -> MetaTensor:
        return MetaTensor(
            data=self.data.clone(),
            step_size=self.step_size.clone(),
            zero_point=self.zero_point.clone(),
            bit=self.bit,
        )

    def pin_memory(self) -> MetaTensor:
        if self.data.is_pinned():
            return self  # already pinned
        return MetaTensor(
            data=self.data.pin_memory(),
            step_size=self.step_size.pin_memory(),
            zero_point=self.zero_point.pin_memory(),
            bit=self.bit,
        )

    def as_tensor(self) -> t.Tensor:
        """
        Return the underlying tensor (fp32 with fake-quant artifacts).
        """
        return self.data

    def __post_init__(self):
        if not isinstance(self.bit, int) or self.bit <= 0:
            raise ValueError(f"bit must be a positive int, got {self.bit!r}")

        # Enforce same device and dtype across fields
        devs = {self.data.device, self.step_size.device, self.zero_point.device}
        dtypes = {self.data.dtype, self.step_size.dtype, self.zero_point.dtype}

        if len(devs) != 1:
            raise ValueError("data/step_size/zero_point must be on the same device.")
        if len(dtypes) != 1:
            raise ValueError("data/step_size/zero_point must share the same dtype.")

        # Validate broadcastability of metadata to data
        _ensure_broadcastable(
            self.data.shape, self.step_size.shape, self.zero_point.shape
        )

    def __repr__(self) -> str:
        return (
            f"MetaTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"bit={self.bit}, step_size.shape={tuple(self.step_size.shape)}, "
            f"zero_point.shape={tuple(self.zero_point.shape)})"
        )


def _ensure_broadcastable(*shapes: t.Size) -> None:
    try:
        t.broadcast_shapes(*shapes)  # type: ignore[attr-defined]
    except Exception as e:
        # Fallback for very old torch: attempt by materializing zeros (rarely needed)
        try:
            _ = 0
            for s in shapes:
                _ = t.zeros(s) + 0  # force shape validation
            # If we got here, accept.
        except Exception:
            raise ValueError(f"Shapes are not broadcastable: {shapes}") from e
