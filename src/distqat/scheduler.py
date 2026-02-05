import torch
# from hivemind.moe.server.layers.lr_schedule import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


schedule_name_to_scheduler = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup, "none": None}
