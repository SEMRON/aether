from transformers import AutoConfig, AutoModelForCTC
import torch
import torch.nn as nn
from hivemind.moe.server.layers.custom_experts import register_expert_class
from pydantic_yaml import parse_yaml_file_as
from distqat.config import Config
from typing import Optional


head_sample_input = lambda batch_size, sampling_rate: (
    torch.randn((batch_size, sampling_rate), dtype=torch.float32),
)

body_sample_input = lambda batch_size, hid_dim: (
    torch.empty((batch_size, 100, hid_dim)),
)

tail_sample_input = lambda batch_size, hid_dim: (
    torch.empty((batch_size, 100, hid_dim)),
)

def get_feat_extract_output_lengths(
    input_lengths: torch.LongTensor,
    config: Optional[AutoConfig] = None,
):
        """
        Computes the output length of the convolutional layers
        """
        config = config or AutoConfig.from_pretrained(full_model_name)

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
        
        for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

class Wav2Vec2(nn.Module):
    def __init__(self, full_model_name: str):
        super().__init__()
        self.model = AutoModelForCTC.from_pretrained(full_model_name)

    def forward(self, x):
        return self.model(x)

@register_expert_class("wav2vec2.full", head_sample_input)
class Wav2Vec2Full(nn.Module):
    def __init__(self, full_model_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(full_model_name)
        self.config.ctc_loss_reduction = "mean"
        self.config.gradient_checkpointing = False
        self.model = AutoModelForCTC.from_config(self.config)

    def forward(self, input_values: torch.Tensor):
        outputs = self.model(input_values)
        logits = outputs.logits
        return logits


@register_expert_class("wav2vec2.head", head_sample_input)
class Wav2Vec2Head(nn.Module):
    def __init__(self, full_model_name: str, n_layers: int = 4):
        super().__init__()
        config = AutoConfig.from_pretrained(full_model_name)
        config.gradient_checkpointing = False
        self.model = AutoModelForCTC.from_config(config)
        self.feature_extractor = self.model.wav2vec2.feature_extractor
        self.feature_extractor._requires_grad = False
        self.feature_projection = self.model.wav2vec2.feature_projection
        full_encoder = self.model.wav2vec2.encoder
        self.blocks = full_encoder.layers[:n_layers]

    def forward(self, input_values: torch.Tensor):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.feature_projection(extract_features)
        for blk in self.blocks:
            hidden_states = blk(hidden_states)[0]
        return hidden_states

    
@register_expert_class("wav2vec2.body", body_sample_input)
class Wav2Vec2Body(nn.Module):
    def __init__(self, full_model_name: str, n_layers: int = 4, idx: int = 4):
        super().__init__()
        config = AutoConfig.from_pretrained(full_model_name)
        config.gradient_checkpointing = False
        self.model = AutoModelForCTC.from_config(config)
        full_encoder = self.model.wav2vec2.encoder
        self.blocks = full_encoder.layers[idx:idx + n_layers]

    def forward(self, hidden_states: torch.Tensor):
        for blk in self.blocks:
            hidden_states = blk(hidden_states)[0]
        return hidden_states


@register_expert_class("wav2vec2.tail", tail_sample_input)
class Wav2Vec2Tail(nn.Module):
    def __init__(self, full_model_name: str, n_layers: int = 4, idx: int = 8):
        super().__init__()
        config = AutoConfig.from_pretrained(full_model_name)
        config.gradient_checkpointing = False
        self.model = AutoModelForCTC.from_config(config)
        full_encoder = self.model.wav2vec2.encoder
        self.blocks = full_encoder.layers[idx:]
        self.dropout = self.model.dropout
        self.lm_head = self.model.lm_head

    def forward(self, hidden_states: torch.Tensor):
        for blk in self.blocks:
            hidden_states = blk(hidden_states)[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits