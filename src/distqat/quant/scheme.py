from __future__ import annotations

from typing import Dict, Any, Union

from pydantic import BaseModel, Field, PositiveInt, model_validator

from .registry import ensure_loaded, get_params_model  # runtime dependency only


class QuantScheme(BaseModel):
    """
    Generic scheme wrapper:
      - algo: string key registered via @register_quantizer in `quantizer/`
      - params: dict or pydantic model; coerced using the registered Params model
    """

    algo: str = "LearnedStepQuantizer"
    num_bits: PositiveInt = 8
    slice_bits: PositiveInt = 4
    params: Union[BaseModel, Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coerce_params(self) -> "QuantScheme":
        ensure_loaded()  # lazy, import-time side effects populate the registry
        pm = get_params_model(self.algo)
        if pm is None:
            raise ValueError(f"Unknown quantizer algo: {self.algo!r}")
        if isinstance(self.params, dict):
            self.params = pm.model_validate(self.params)
        elif not isinstance(self.params, pm):
            self.params = pm.model_validate(self.params.model_dump())
        return self

    def as_factory_kwargs(self) -> Dict[str, Any]:
        out = {
            "algo": self.algo,
            "num_bits": self.num_bits,
            "slice_bits": self.slice_bits,
        }
        out.update(self.params.model_dump())
        return out
