from typing import Dict, Tuple, List, Callable, Optional, Union
import torch as t

from distqat.func import DefaultQuantizedModuleMapping, QUANT_MODULE_MAPPING_TYPE, DqLinear, DqConv2d
from distqat.quant import Quantizer, PassThroughQuantizer
from distqat.quant.scheme import QuantScheme
from distqat.quant.registry import get_spec, ensure_loaded

from distqat.config import QuantConfig
from distqat.utils.rules import OverrideRule, glob_to_regex  # for strict_overrides check


def make_quantizer(scheme: QuantScheme) -> Quantizer:
    """
    Instantiate a quantizer from a QuantScheme (algo + typed params from registry).

    - Ensures decorator registry is loaded.
    - Flattens scheme into kwargs expected by the quantizer ctor.
    - Falls back to IdentityQuantizer if algo isn't registered (shouldn't happen
      if config validated, but kept for safety) or if you explicitly use "Identity".
    """
    ensure_loaded()
    algo = scheme.algo

    # Optional: treat these as identity without requiring registration.
    if algo.lower() in {"identity", "none", "noop"}:
        return PassThroughQuantizer()

    spec = get_spec(algo)
    if spec is None:
        # Config validation should have caught this; be defensive at runtime.
        return PassThroughQuantizer()

    kwargs = scheme.as_factory_kwargs()
    kwargs.pop("algo", None)  # ctor usually doesn't accept 'algo' itself
    return spec.ctor(**kwargs)  # type: ignore[call-arg]


def _replace_module_by_names(
    model: t.nn.Module,
    modules_to_replace: Dict[str, t.nn.Module],
    quantized_module_mapping: QUANT_MODULE_MAPPING_TYPE,
) -> t.nn.Module:
    """
    Replace modules in a model by matching their full names.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose modules will be replaced.
    modules_to_replace : dict of str to torch.nn.Module
        Mapping from module full names to replacement modules.
    quantized_module_mapping : dict
        Mapping from original module classes to quantized module wrapper classes.

    Returns
    -------
    torch.nn.Module
        Model with specified modules replaced.
    """
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in quantized_module_mapping.keys():
                for full_name, m in model.named_modules():
                    if c is m and full_name in modules_to_replace:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model


def _rule_matches_name_and_class(
    rule: OverrideRule, module_path: str, module_cls_name: str
) -> bool:
    """
    Mirror the compiled rule matcher for the strict_overrides precheck,
    without requiring a compiled resolver.
    """
    pats = rule.name_patterns or ["*"]
    regexes = [glob_to_regex(p) for p in pats]
    name_ok = any(rx.fullmatch(module_path) for rx in regexes)
    class_ok = (not rule.class_names) or (module_cls_name in rule.class_names)
    return name_ok and class_ok


def attach_quantizers(
    model: t.nn.Module,
    quan_cfg: QuantConfig,
    quantized_module_mapping: QUANT_MODULE_MAPPING_TYPE = DefaultQuantizedModuleMapping,
    strict_overrides: bool = True,
    replace_if_identity: bool = True,
) -> t.nn.Module:
    """
    Attach quantizers to modules in a model according to a quantization configuration.

    This function traverses the model, computes effective per-layer quantization
    configurations via :meth:`QuantConfig.resolve_for`, and replaces modules with
    their quantized counterparts defined in ``quantized_module_mapping``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to which quantizers will be attached.
    quan_cfg : QuantConfig
        Global quantization configuration containing default and override settings.
    quantized_module_mapping : dict, optional
        Mapping from original module types to their quantized wrapper classes.
        Default is ``DefaultQuantizedModuleMapping``.
    strict_overrides : bool, optional
        If True, raises an error when an override matches a non-quantizable module.
        Default is True.
    replace_if_identity : bool, optional
        If False, skips replacing modules when both weight and activation quantizers
        are identity quantizers. Default is True.

    Returns
    -------
    torch.nn.Module
        The model with quantized modules attached.

    Raises
    ------
    KeyError
        If ``strict_overrides`` is True and an override matches a non-quantizable module.

    Notes
    -----
    - Quantizers are created using :func:`make_quantizer`.
    - If the wrapper class supports accumulator arguments, they are passed through;
      otherwise, they are omitted.
    """
    replaceable_types = set(quantized_module_mapping.keys())

    # Guardrail: if rules target modules we can't replace, surface it early.
    if strict_overrides and getattr(quan_cfg, "overrides", None):
        non_replaceable_hits: List[str] = []
        module_index = {
            name: (mod.__class__, mod.__class__.__name__)
            for name, mod in model.named_modules()
        }
        for rule in quan_cfg.overrides:
            for name, (cls, cls_name) in module_index.items():
                if _rule_matches_name_and_class(rule, name, cls_name) and cls not in replaceable_types:
                    non_replaceable_hits.append(name)
        if non_replaceable_hits:
            unique = sorted(set(non_replaceable_hits))
            raise KeyError(f"Overrides matched non-quantizable modules: {unique}")

    modules_to_replace: Dict[str, t.nn.Module] = {}

    for name, module in model.named_modules():
        cls = type(module)
        if cls in replaceable_types:
            # Resolve effective per-layer config (weights, activations, accumulators)
            eff = quan_cfg.resolve_for(module_path=name, module_cls_name=module.__class__.__name__)

            # Instantiate quantizers via the new scheme+registry path
            qw = make_quantizer(eff.weight_quant)
            qa = make_quantizer(eff.activation_quant)

            if (
                not replace_if_identity
                and isinstance(qw, PassThroughQuantizer)
                and isinstance(qa, PassThroughQuantizer)
            ):
                continue

            mapped_module_cls = quantized_module_mapping[cls]
            # Pass accumulator args if wrapper supports them; fall back otherwise
            try:
                wrapped = mapped_module_cls(
                    module,
                    w_quant_fn=qw,
                    x_quant_fn=qa,
                    accumulator_length=eff.accumulator_length,
                    accumulator_bits=eff.accumulator_bits,
                )
            except TypeError:
                wrapped = mapped_module_cls(module, w_quant_fn=qw, x_quant_fn=qa)

            modules_to_replace[name] = wrapped

    transformed_module = _replace_module_by_names(model, modules_to_replace, quantized_module_mapping)

    quantization_params = []
    to_be_averaged_module_prefixes = []
    for module_name, module in transformed_module.named_modules():
        if any(module_name.startswith(ignore) for ignore in to_be_averaged_module_prefixes):
            quantization_params.extend(module.parameters())
            continue
        
        if isinstance(module, (DqLinear, DqConv2d)):
            to_be_averaged_module_prefixes.append(module_name)
            continue

    return transformed_module, quantization_params