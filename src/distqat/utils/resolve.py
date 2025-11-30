from __future__ import annotations

from typing import Generic, Iterable, List, Optional, Tuple, TypeVar

import functools, re

from pydantic import BaseModel
from .rules import OverrideRule, glob_to_regex, specificity


__all__ = [
    "CompiledResolver",
    "compile_overrides",
]

C = TypeVar("C", bound=BaseModel)  # concrete (full) config model
P = TypeVar("P", bound=BaseModel)  # partial payload model


class _CompiledRule(BaseModel, Generic[P]):
    regexes: List[re.Pattern]
    classes: set[str]
    payload: P
    spec: Tuple[int, int]  # specificity

    @classmethod
    def from_rule(cls, r: OverrideRule[P]) -> "_CompiledRule[P]":
        pats = r.name_patterns or ["*"]
        regs = [glob_to_regex(p) for p in pats]
        spec = max((specificity(p) for p in pats), default=(0, 0))
        return cls(
            regexes=regs, classes=set(r.class_names), payload=r.payload, spec=spec
        )

    def matches(self, module_path: str, module_cls: str) -> bool:
        name_ok = any(rx.fullmatch(module_path) for rx in self.regexes)
        class_ok = (not self.classes) or (module_cls in self.classes)
        return name_ok and class_ok


class CompiledResolver(Generic[C, P]):
    """
    Applies a sequence of OverrideRule[P] overlays to produce an effective config C.

    Merge semantics are shallow field overlays: for each matched rule we
    model_dump(exclude_none=True) the payload and model_copy(update=...) the base.
    """

    def __init__(self, base: C, compiled_rules: List[_CompiledRule[P]]):
        self._base = base
        # Order: less specific first, more specific last → "last wins" favors specificity
        self._rules = sorted(compiled_rules, key=lambda r: (r.spec[0], r.spec[1]))

    @functools.lru_cache(maxsize=8192)
    def resolve_for(self, module_path: str, module_cls_name: str) -> C:
        eff = self._base
        for cr in self._rules:
            if cr.matches(module_path, module_cls_name):
                eff = _merge(eff, cr.payload)
        return eff


def compile_overrides(
    base: C, overrides: Iterable[OverrideRule[P]]
) -> CompiledResolver[C, P]:
    compiled = [_CompiledRule.from_rule(r) for r in overrides]
    return CompiledResolver(base, compiled)


def _merge(base: C, payload: P) -> C:
    """
    Shallow overlay with validation: we rebuild the model so nested fields
    (e.g., QuantScheme) are coerced from dicts into proper models.
    """
    upd = payload.model_dump(exclude_none=True)
    if not upd:
        return base

    # filter unknown keys to avoid surprises
    allowed = set(base.model_fields.keys())
    clean = {k: v for k, v in upd.items() if k in allowed}

    # Rebuild with validation to coerce nested dicts into submodels
    merged = base.model_dump()
    merged.update(clean)
    return type(base).model_validate(merged)
