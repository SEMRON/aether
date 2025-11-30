from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel, PositiveInt

import torch as t

from .meta_tensor import MetaTensor


# class QuantizerKwargs(BaseModel):
#     num_bits: PositiveInt = 8
#     slice_bits: PositiveInt = 4


class Quantizer(ABC, t.nn.Module):
    """
    Abstract base-class for *fake* quantisers working with ``MetaTensor``.

    Forward routine *always* returns an ``MetaTensor`` in fp32 with
    quantisation artefacts baked in.
    """

    is_initialized: t.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("is_initialized", t.tensor(0, dtype=t.bool))

    def forward(self, x: t.Tensor) -> MetaTensor:
        if not bool(self.is_initialized):
            self.initialize(x)
            self.is_initialized.fill_(True)

        return self.fake_quant(x)

    @abstractmethod
    def initialize(self, x: t.Tensor) -> None:
        """Populate quant-params (e.g. step size)."""
        raise NotImplementedError

    @abstractmethod
    def fake_quant(self, x: t.Tensor) -> MetaTensor:
        """Return ``MetaTensor`` whose data == dequant(quant(x))."""
        raise NotImplementedError
    
    def extra_repr(self) -> str:
        bits = getattr(self, "bit", None)
        parts = []
        if bits is not None:
            parts.append(f"bit={bits}")
        parts.append(f"init={bool(self.is_initialized)}")
        return ", ".join(parts)
