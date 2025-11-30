from __future__ import annotations

from typing import Dict, Type, Callable, Optional, Iterable, Any

import inspect
import pkgutil
import sys

from dataclasses import dataclass
from importlib import import_module
from pydantic import BaseModel


@dataclass(frozen=True)
class QuantizerSpec:
    params_model: Type[BaseModel]  # pydantic model for algo-specific params
    ctor: Type[Any]  # the quantizer class itself (implementation)
    aliases: tuple[str, ...] = ()  # optional string aliases


_REGISTRY: Dict[str, QuantizerSpec] = {}
_LOADED: bool = False


def register_quantizer(
    name: Optional[str] = None, *, aliases: Iterable[str] = ()
) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register a quantizer implementation class.

    The class must expose a pydantic params model via either:
      - `Params` attribute, or
      - `ParamsModel` attribute.
    """

    def _decorator(cls: Type[Any]) -> Type[Any]:
        nm = (name or getattr(cls, "name", cls.__name__)).strip()
        params_model = getattr(cls, "Params", None) or getattr(cls, "ParamsModel", None)
        if not (inspect.isclass(params_model) and issubclass(params_model, BaseModel)):
            raise TypeError(f"{cls.__name__} must define `Params` = pydantic.BaseModel")

        spec = QuantizerSpec(
            params_model=params_model, ctor=cls, aliases=tuple(aliases)
        )
        # main name
        _REGISTRY[nm] = spec
        # aliases (canonicalize to str)
        for a in aliases:
            _REGISTRY[str(a)] = spec
        return cls

    return _decorator


def get_spec(name: str) -> Optional[QuantizerSpec]:
    return _REGISTRY.get(name)


def get_params_model(name: str) -> Optional[Type[BaseModel]]:
    spec = get_spec(name)
    return spec.params_model if spec else None


def names() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def clear_registry() -> None:
    """Useful for tests/hot-reload."""
    _REGISTRY.clear()


def load_all() -> None:
    """
    Import all submodules of the `quantizer` package so decorators run.
    Safe to call multiple times; no-ops after first.
    """
    global _LOADED
    if _LOADED:
        return
    pkg_name = __name__.rsplit(".", 1)[0]  # 'quantizer'
    pkg = import_module(pkg_name)
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        # Skip registry itself to avoid re-entry
        if modname.endswith(".registry"):
            continue
        if modname in sys.modules:
            continue
        import_module(modname)
    _LOADED = True


def ensure_loaded() -> None:
    if not _LOADED:
        load_all()
