from __future__ import annotations
import math

import torch as t
from pydantic import BaseModel

from distqat.utils.quantizer import (
    calculate_bounds,
    grad_scale,
    round_pass,
)
from ..base import Quantizer
from ..meta_tensor import MetaTensor
from ..registry import register_quantizer


class LSQParams(BaseModel):
    pass


@register_quantizer("LearnedStepQuantizer", aliases=["lsq"])
class LearnedStepQuantizer(Quantizer):
    """
    LSQ quantiser (Esser *et al.*, 2019) producing fake-quantised outputs.

    References
    ----------
    Steven K. Esser, et al. "LSQ: Learned Step Size Quantization." ICLR 2020.
    """
    Params = LSQParams

    upper_bound: t.Tensor
    lower_bound: t.Tensor
    step_size: t.nn.Parameter

    def __init__(self, num_bits: int, *args, **kwargs) -> None:
        super().__init__()

        self.bit = num_bits
        ub, lb = calculate_bounds(
            num_bits, False, False
        )  # unsigned: bool = False, symmetric: bool = False
        self.register_buffer("upper_bound", t.tensor(ub))
        self.register_buffer("lower_bound", t.tensor(lb))

        # learnable step size (initialised later)
        self.step_size = t.nn.Parameter(t.tensor(1.0))

    def initialize(self, x: t.Tensor) -> None:
        if bool(self.is_initialized):
            return

        mean_abs = x.detach().abs().mean()

        # The ExpertBackends send zero values in the dummy forward pass
        # which would lead to miss-initialized quantizer step_size
        threshold = 1e-6
        if mean_abs <= threshold:
            print(
                f"NOTE: Skipping initialization of LearnedStepQuantizer, mean_abs <= {threshold}. "
                f"This is expected for the dummy forward pass from the ExpertBackends."
            )
            return

        self.step_size.data.copy_(2 * mean_abs / math.sqrt(self.upper_bound))

    def fake_quant(self, x: t.Tensor) -> MetaTensor:
        grad_scale_factor = t.rsqrt(x.numel() * self.upper_bound)
        scaled_step = grad_scale(self.step_size, grad_scale_factor)

        # integer quantisation with STE rounding + clamping
        q_int = round_pass(t.clamp(x / scaled_step, self.lower_bound, self.upper_bound))

        # de‑quantise to obtain fake‑quant fp32 values
        x_fake = q_int * scaled_step

        return MetaTensor.from_tensor(
            x_fake,
            step_size=scaled_step,
            bit=self.bit,
        )
