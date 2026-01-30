from distqat.config import Config
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig
from hivemind.utils.logging import get_logger

from distqat.models.wav2vec2 import get_feat_extract_output_lengths
from distqat.utils.metrics import MetricsLogger

logger = get_logger(__name__)
logger.setLevel('DEBUG')

# Cache for AutoConfig to avoid repeated loading and deprecation warnings
_config_cache: dict[str, AutoConfig] = {}

def _get_cached_config(model_name: str) -> AutoConfig:
    """Get a cached AutoConfig, loading it only once per model name."""
    if model_name not in _config_cache:
        cfg = AutoConfig.from_pretrained(model_name)
        # Remove deprecated attribute to suppress warnings on future use
        if hasattr(cfg, 'gradient_checkpointing'):
            delattr(cfg, 'gradient_checkpointing')
        _config_cache[model_name] = cfg
    return _config_cache[model_name]


def task_type_loss(config: Config, inputs, outputs, labels, metrics_logger: MetricsLogger = None, step=None):
        task_type = config.data.task_type
        if task_type == "cv":
            return F.cross_entropy(outputs.float(), labels.to(outputs.device))
        elif task_type == "llm":
            return outputs.mean()
        elif task_type == "speech":
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
            model_name = config.data.full_model_name
            input_lengths = get_feat_extract_output_lengths(attention_mask.sum(-1), config=_get_cached_config(model_name)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = F.log_softmax(outputs, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    reduction="mean",
                )
            return loss
        elif task_type == "image_gen":
            logger.warning("Image generation loss is implemented separately in the model")
            return None
        elif task_type == "node_pred":
            loss = outputs.squeeze(0)[0]
            return loss
        elif task_type == "rl":
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = inputs

            outputs = tuple(item.to(config.device) if hasattr(item, "to") else item for item in outputs)
            _, newlogprob, entropy, newvalue = outputs
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_coef = float(config.ppo.clip_coef)
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

            mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - b_returns) ** 2
            v_clipped = b_values + torch.clamp(newvalue - b_values, -clip_coef, clip_coef)
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - float(config.ppo.ent_coef) * entropy_loss + v_loss * float(config.ppo.vf_coef)

            # logging
            y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            log_scalar = metrics_logger.log_scalar if metrics_logger is not None else None
            if log_scalar is not None:
                log_scalar("losses/value_loss", v_loss.item(), step)
                log_scalar("losses/policy_loss", pg_loss.item(), step)
                log_scalar("losses/entropy", entropy_loss.item(), step)
                log_scalar("losses/old_approx_kl", old_approx_kl.item(), step)
                log_scalar("losses/approx_kl", approx_kl.item(), step)
                log_scalar("losses/clipfrac", clipfrac, step)
                log_scalar("losses/explained_variance", float(explained_var), step)
                log_scalar("charts/global_step", step, step)

            return loss
        else:
            raise ValueError(f"Unknown task type: {task_type}")