from typing import Dict, Type

import torch as t

from .layer import DqLinear, DqConv2d


QUANT_MODULE_MAPPING_TYPE = Dict[Type[t.nn.Module], Type[t.nn.Module]]
DefaultQuantizedModuleMapping: QUANT_MODULE_MAPPING_TYPE = {
    t.nn.Conv2d: DqConv2d,
    t.nn.Linear: DqLinear,
}

__all__ = ["DqLinear", "DqConv2d"]
