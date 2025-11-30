from __future__ import annotations

from typing import Optional

import math

import torch as t
import torch.nn.functional as F

from distqat.quant import Quantizer, MetaTensor


class DqLinear(t.nn.Linear):
    """
    Linear with injected quantizers.
    """

    def __init__(
        self,
        m: t.nn.Linear,
        x_quant_fn: Optional[Quantizer] = None,
        w_quant_fn: Optional[Quantizer] = None,
        accumulator_length: int = 512,
        accumulator_bits: int = 8,
    ) -> None:
        assert type(m) == t.nn.Linear
        super().__init__(
            in_features=m.in_features,
            out_features=m.out_features,
            bias=True if m.bias is not None else False,
        )
        self.x_quant_fn = x_quant_fn
        self.w_quant_fn = w_quant_fn

        self.weight = t.nn.Parameter(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_meta: MetaTensor = self.x_quant_fn(x)
        w_meta: MetaTensor = self.w_quant_fn(self.weight)

        return F.linear(x_meta.data, w_meta.data, bias=self.bias)


class DqConv2d(t.nn.Conv2d):
    """
    Conv2d with injected quantizers.
    """

    def __init__(
        self,
        m: t.nn.Conv2d,
        x_quant_fn: Optional[Quantizer] = None,
        w_quant_fn: Optional[Quantizer] = None,
        accumulator_length: int = 512,
        accumulator_bits: int = 8,
    ) -> None:
        assert type(m) == t.nn.Conv2d
        super().__init__(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode,
        )
        self.x_quant_fn = x_quant_fn
        self.w_quant_fn = w_quant_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_meta: MetaTensor = self.x_quant_fn(x)
        w_meta: MetaTensor = self.w_quant_fn(self.weight)

        return self._conv_forward(x_meta.data, w_meta.data, bias=self.bias)
