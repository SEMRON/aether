from __future__ import annotations

import torch as t
from pydantic import BaseModel

from ..base import Quantizer
from ..meta_tensor import MetaTensor
from ..registry import register_quantizer


class PassThroughParams(BaseModel):
    pass

@register_quantizer("PassThroughQuantizer", aliases=["passthrough", "identity"])
class PassThroughQuantizer(Quantizer):
    """
    A quantizer that passes through the input tensor without any quantization.
    """
    Params = PassThroughParams

    def __init__(self):
        super().__init__()
        self.register_buffer("step_size", t.ones(()))

    def initialize(self, x: t.Tensor):
        pass  # step_size already 1.0

    def fake_quant(self, x: t.Tensor) -> MetaTensor:
        return MetaTensor.from_tensor(x, stepsize=self.step_size, bit=32)
