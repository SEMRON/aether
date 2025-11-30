from __future__ import annotations

import fnmatch, re

from typing import Generic, List, Tuple, TypeVar
from pydantic import BaseModel, Field

__all__ = [
    "glob_to_regex",
    "specificity",
    "OverrideRule",
]

P = TypeVar("P", bound=BaseModel)  # payload model (usually a *partial* config)


def glob_to_regex(pat: str) -> re.Pattern:
    """Compile a glob pattern (fnmatch) into a regex pattern with fullmatch semantics."""
    return re.compile(fnmatch.translate(pat))


def specificity(pat: str) -> Tuple[int, int]:
    """
    Specificity score for ordering rules:
      - fewer wildcards → more specific
      - more segments (dot-separated) → more specific
    Higher tuple values are "more specific".
    """
    wildcards = pat.count("*") + pat.count("?")
    segments = pat.count(".") + 1
    return (-wildcards, segments)


class OverrideRule(BaseModel, Generic[P]):
    """
    A rule that applies a partial payload to items selected by:
      - name_patterns: glob(s) against a qualified module path (e.g., 'encoder.layers.*.self_attn')
      - class_names: set of class-name strings (empty means 'any class')

    'payload' should be a Pydantic model whose fields are Optional[...] so that
    unset fields (None) do not override the base config. Downstream, we merge with
    model_dump(exclude_none=True).
    """

    name_patterns: List[str] = Field(default_factory=list)
    class_names: List[str] = Field(default_factory=list)
    payload: P
