from distqat.config import Config, ModelConfig, DataConfig

from typing import Callable, Dict, Any, Type
from inspect import signature

from .resnet import (
    ResNet18Full, 
    Resnet18Front, 
    Resnet18Back,
    ResNet50Full,
    ResNet101Full,
)
from .mlp import MLP, MLPFront, MLPBack
from .distilgpt2 import (
    DistilGPT2Full,
    DistilGPT2Head,
    DistilGPT2Body,
    DistilGPT2Tail,
)
from .gpt_neo import GPTNeoFull, GPTNeoHeadExpert, GPTNeoBodyExpert, GPTNeoTailExpert
from .wav2vec2 import Wav2Vec2Full, Wav2Vec2Head, Wav2Vec2Body, Wav2Vec2Tail
from .biggan.biggan_adapter import BigGANAdapter

MODEL_TYPES: Dict[str, Type[Any]] = {
    "mlp": MLP,
    "mlp.head": MLPFront,
    "mlp.tail": MLPBack,
    "resnet18.full": ResNet18Full,
    "resnet18.head": Resnet18Front,
    "resnet18.tail": Resnet18Back,
    "resnet50.full": ResNet50Full,
    "resnet101.full": ResNet101Full,
    "distilgpt2.full": DistilGPT2Full,
    "distilgpt2.head": DistilGPT2Head,
    "distilgpt2.body": DistilGPT2Body,
    "distilgpt2.tail": DistilGPT2Tail,
    "gptneo.full": GPTNeoFull,
    "gptneo.head": GPTNeoHeadExpert,
    "gptneo.body": GPTNeoBodyExpert,
    "gptneo.tail": GPTNeoTailExpert,
    "wav2vec2.full": Wav2Vec2Full,
    "wav2vec2.head": Wav2Vec2Head,
    "wav2vec2.body": Wav2Vec2Body,
    "wav2vec2.tail": Wav2Vec2Tail,
    "biggan.full": BigGANAdapter,
}


def kwargs_from_config(fn: Callable, model_config: ModelConfig, data_config: DataConfig, aliases: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Build kwargs for `fn` from both model_cfg and data_cfg.
    - Merges data_cfg into model_cfg (data overrides model on conflicts)
    - Optional aliases let you compute/rename fields (e.g. {"img_size": data_cfg.img_size})
    """
    md = model_config.model_dump(exclude_unset=True)
    dd = data_config.model_dump(exclude_unset=True)

    merged = {**dd, **md}

    if aliases:
        merged.update(aliases)

    params = signature(fn).parameters
    return {name: merged[name] for name in params if name != "self" and name in merged}


def get_model(config: Config, pipeline_step_cfg: ModelConfig) -> Any:
    try:
        model_cls = MODEL_TYPES[pipeline_step_cfg.model_name]
    except KeyError:
        # if pipeline_step_cfg.model_name == "biggan.full":
        #     raise ValueError("BigGAN is not supported in the baseline model")
        available = ", ".join(sorted(MODEL_TYPES.keys()))
        raise ValueError(f"Unknown model '{pipeline_step_cfg.model_name}'. Available: {available}")
    
    aliases = {
        "config": config.biggan,
    }
    kwargs = kwargs_from_config(model_cls.__init__, pipeline_step_cfg, config.data, aliases)
    return model_cls(**kwargs)
