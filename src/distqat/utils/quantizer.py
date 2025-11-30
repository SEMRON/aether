from typing import Tuple

import torch as t


def calculate_bounds(
    bit: int, unsigned: bool = False, symmetric: bool = True
) -> Tuple[int, int]:
    if unsigned:
        assert not symmetric, "Unsigned quantization cannot be symmetric"
        # unsigned activation is quantized to [0, 2^b-1]
        lower_bound = 0
        upper_bound = 2**bit - 1
    else:
        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            lower_bound = -(2 ** (bit - 1)) + 1
            upper_bound = 2 ** (bit - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            lower_bound = -(2 ** (bit - 1))
            upper_bound = 2 ** (bit - 1) - 1

    return upper_bound, lower_bound


def grad_scale(x: t.Tensor, scale: t.Tensor) -> t.Tensor:
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x: t.Tensor) -> t.Tensor:
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad
