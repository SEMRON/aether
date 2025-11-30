from transformers import AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn
from hivemind.moe.server.layers.custom_experts import register_expert_class
from pydantic_yaml import parse_yaml_file_as
from distqat.config import Config



head_sample_input = lambda batch_size, seq_len: (
    torch.randint(low=0, high=1000, size=(batch_size, seq_len), dtype=torch.long),
)

body_sample_input = lambda batch_size, hid_dim, seq_len: (
    torch.empty((batch_size, seq_len, hid_dim)),
)

tail_sample_input = lambda batch_size, hid_dim, seq_len: (
    torch.empty((batch_size, seq_len, hid_dim)),
)

class GPTNeo(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_config(config)
        
    def forward(self, x):
        logits = self.model(input_ids=x, return_dict=False)[0]
        return logits

# ---------- Full: full model ----------
# @register_expert_class("gptneo.full", lambda batch_size, hid_dim: torch.empty((batch_size, 128), dtype=torch.long))
@register_expert_class("gptneo.full", head_sample_input)
class GPTNeoFull(nn.Module):
    def __init__(self, full_model_name: str):
        super(GPTNeoFull, self).__init__()
        self.model = GPTNeo(full_model_name)

    def forward(self, x):
        return self.model(x)

# ---------- Head: embeddings + first L_head blocks ----------
@register_expert_class("gptneo.head", head_sample_input)
class GPTNeoHeadExpert(nn.Module):
    def __init__(self, full_model_name: str, hid_dim: int, n_layers: int = 8):
        super().__init__()
        print(hid_dim)
        print(n_layers)
        self.n_layers = n_layers
        cfg = AutoConfig.from_pretrained(full_model_name)

        full = AutoModelForCausalLM.from_config(cfg)
        tr = full.transformer
        self.wte, self.wpe, self.drop = tr.wte, tr.wpe, tr.drop     
        self.blocks = nn.ModuleList(tr.h[:n_layers])

    def forward(self, input_ids: torch.LongTensor):
        # input_ids: [B, T]
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        hidden = self.wte(input_ids) + self.wpe(pos)
        hidden = self.drop(hidden)
        # No pad mask -> pure causal attention inside the blocks (TODO: check if we need mask for packing?)
        for blk in self.blocks:
            hidden = blk(hidden, attention_mask=None)[0]
        return hidden

# ---------- Body: next L_body blocks ----------
@register_expert_class("gptneo.body", body_sample_input)
class GPTNeoBodyExpert(nn.Module):
    def __init__(self, full_model_name: str, hid_dim: int, n_layers: int = 8, idx: int = 8):
        super().__init__()
        self.n_layers, self.idx = n_layers, idx
        cfg = AutoConfig.from_pretrained(full_model_name)
        full = AutoModelForCausalLM.from_config(cfg)
        tr = full.transformer
        self.blocks = nn.ModuleList(tr.h[idx:idx + n_layers])

    def forward(self, hidden_states: torch.Tensor):
        for blk in self.blocks:
            hidden_states = blk(hidden_states, attention_mask=None)[0]
        return hidden_states

# ---------- Tail: last L_tail blocks + ln_f + lm_head ----------
@register_expert_class("gptneo.tail", tail_sample_input)
class GPTNeoTailExpert(nn.Module):
    def __init__(self, full_model_name: str, hid_dim: int, n_layers: int = 8, idx: int = 8):
        super().__init__()
        self.n_layers, self.idx = n_layers, idx
        cfg = AutoConfig.from_pretrained(full_model_name)
        full = AutoModelForCausalLM.from_config(cfg)
        tr = full.transformer
        self.blocks = nn.ModuleList(tr.h[idx:])
        self.ln_f = tr.ln_f
        self.lm_head = full.lm_head

    def forward(self, hidden_states: torch.Tensor):
        for blk in self.blocks:
            hidden_states = blk(hidden_states, attention_mask=None)[0]
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
