import torch
torch.multiprocessing.set_sharing_strategy('file_descriptor')
import torch.nn.functional as F

import signal
import warnings
import time
from transformers import AutoConfig

import click
from hivemind.utils.logging import get_logger
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import SchemaValidator

from distqat.distributed.model import SwarmBaselineModel
from distqat.distributed.model import SwarmModel, SwarmBaselineModel
from distqat.distributed.optim.diloco import TrainingProgressSchema
from distqat.config import Config, parse_args
from distqat.distributed.optim.diloco import TrainingState
from distqat.sharding import register_process, get_outer_step, set_outer_step_if_greater
from distqat.models.wav2vec2 import get_feat_extract_output_lengths
from distqat.distributed.data_client import DataClient


logger = get_logger(__name__)
logger.setLevel('DEBUG')

class SwarmTrainer:
    """
    A trainer for distributed/collaborative training using hivemind swarm.
    
    This trainer can operate in two modes:
    1. Distributed mode: Uses remote experts for model computation via SwarmModel
    2. Baseline mode: Uses local model computation via SwarmBaselineModel
    
    Args:
        trainer_id: Unique identifier for this trainer instance
        config: Configuration containing data, model, network, and training parameters
        use_baseline_model: If True, uses local computation instead of remote experts
    """
    def __init__(self,
        trainer_id: int,
        config: Config,
        use_baseline_model: bool = False,
        disable_quant: bool = False,
    ):
        self.config = config
        self.trainer_id = trainer_id
        self.run_id = config.experiment_prefix + "_" + str(trainer_id)

        self.device = config.device
        
        self.num_warmup_steps = 0
        self.num_total_steps = config.diloco.total_steps

        self.use_baseline_model = use_baseline_model
        if not self.use_baseline_model:
            self.model = SwarmModel(
                config=self.config,
                trainer_id=self.trainer_id,
            )
        else:
            self.model = SwarmBaselineModel(
                config=self.config,
                trainer_id=self.trainer_id,
                disable_quant=disable_quant,
            )
    

        self.process_id = self.model.dht.peer_id.to_string()
        register_process(self.model.dht, self.config.experiment_prefix, ttl=300.0)
        self.shm_client = DataClient(host=self.config.data_server.ipc_host, port=int(self.config.data_server.ipc_port), authkey=self.config.data_server.ipc_key.encode())

        self.batch_size = self.config.diloco.batch_size_per_step
        self.remaining_batch = None
        self.remaining_release = None

    def parameters(self):
        yield from self.model.parameters()


    def task_type_loss(self, inputs, outputs, labels):
        if self.config.data.task_type == "cv":
            return F.cross_entropy(outputs, labels)
        elif self.config.data.task_type == "llm":
            # shift for next-token prediction
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.permute(0, 2, 1), 
                shift_labels,
                reduction="none",
            )
            loss = loss.mean()
            return loss
        elif self.config.data.task_type == "speech":
            attention_mask = torch.ones_like(inputs, dtype=torch.long)
            model_name = self.config.data.full_model_name
            input_lengths = get_feat_extract_output_lengths(attention_mask.sum(-1), config=AutoConfig.from_pretrained(model_name)).to(torch.long)

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
        else:
            raise ValueError(f"Unknown task type: {self.config.data.task_type}")


    def step(self, inner_step: int, step: int):
        inputs_list = []
        labels_list = []
        releases_to_call = []
        
        samples_needed = self.batch_size
        
        while samples_needed > 0:
            if self.remaining_batch is None:
                self.remaining_batch, self.remaining_release = self.shm_client.next_batch()
            
            current_inputs = self.remaining_batch["inputs"]
            current_labels = self.remaining_batch["labels"]
            available = current_inputs.shape[0]
            
            take = min(samples_needed, available)
            
            inputs_list.append(current_inputs[:take])
            labels_list.append(current_labels[:take])
            
            samples_needed -= take
            
            if take == available:
                releases_to_call.append(self.remaining_release)
                self.remaining_batch = None
                self.remaining_release = None
            else:
                self.remaining_batch["inputs"] = current_inputs[take:]
                self.remaining_batch["labels"] = current_labels[take:]
        
        inputs = torch.cat(inputs_list)
        labels = torch.cat(labels_list)
        
        for release in releases_to_call:
            try:
                release()
            except Exception:
                pass
        
        if inner_step % 10 == 0:
            logger.info(f"Inner step {inner_step} of {self.config.diloco.inner_steps}")
            logger.info(f"Batch size: {inputs.shape}")
        
        outputs = self.model(inputs)

        loss = self.task_type_loss(inputs, outputs, labels)
        loss.backward()
        self.model.post_optimizer_callback(step, loss.item())

    def train(self):
        logger.info(f"============= Training for {self.num_total_steps} steps =============")

        inner_steps = self.config.diloco.inner_steps
        outer_step = 0
        prefix = self.config.experiment_prefix
        dht = self.model.dht 
        signature_validator = RSASignatureValidator()
        dht.add_validators([SchemaValidator(TrainingProgressSchema, prefix=prefix), signature_validator])
       
        step = 0

        logger.info(f"============= Checking Progress of other Servers =============")
        while True:
            max_outer_step_progress_ratio = 0.0
            n_stages = len(self.config.model_pipeline.pipeline)
            for stage_index in range(n_stages):
                summed_inner_steps, n_servers = 0, 0

                progress_dict = dht.get(f"{prefix}_{stage_index}_progress", latest=True)
                if progress_dict is not None:
                    progress_dict = progress_dict.value
                    progress = [
                        TrainingState.validate(entry.value)
                        for entry in progress_dict.values()
                        if entry.value is not None
                    ]
                    summed_inner_steps += sum(p.inner_step for p in progress)
                    n_servers = len(progress)
                    if n_servers > 0 and inner_steps > 0:
                        outer_step_progress_ratio = summed_inner_steps / (n_servers * inner_steps)
                    else:
                        outer_step_progress_ratio = 0.0
                else:
                    outer_step_progress_ratio = 0.0

                max_outer_step_progress_ratio = max(
                    max_outer_step_progress_ratio,
                    outer_step_progress_ratio,
                )

            if max_outer_step_progress_ratio < 0.2:
                break

            logger.info(f"Waiting for other servers to progress... (ratio: {max_outer_step_progress_ratio})")
            time.sleep(1.0)

        logger.info(f"============= Starting training =============")

        # TODO: Flatten the loops back to while not done and add the rebuild_pipeline step with a modulo step based on a config parameter
        while step < self.num_total_steps:
            # heartbeat into active-set with TTL if you have a refresher
            register_process(dht, prefix, ttl=300.0)
            
            # (Re)build sharded pipeline for this outer step 
            shared = get_outer_step(dht, prefix)
            outer_step = max(outer_step, shared)
            
            # Inner loop: exactly inner_steps batches for this peer
            inner = 0

            while inner < inner_steps:
                self.step(inner_step=inner, step=step)
                
                step += 1
                inner += 1
                if step >= self.num_total_steps:
                    break

            set_outer_step_if_greater(dht, prefix, outer_step + 1)

            outer_step += 1

        logger.info(f"============= Training finished =============")

    def shutdown(self):
        self.model.shutdown()


def main(cfg: Config, trainer_id: int, run_locally: bool, disable_quant: bool = False):
    trainer = SwarmTrainer(
        trainer_id=trainer_id,
        config=cfg,
        use_baseline_model=run_locally,
        disable_quant=disable_quant,
    )
    signal.signal(signal.SIGINT, signal.default_int_handler)
    
    logger.info(f"Created SwarmTrainer with config from {cfg}")
    logger.info(f"Trainer ID: {trainer_id}")
    logger.info(f"Using baseline model: {run_locally}")

    try:    
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error in trainer: {e}")
        raise e
    finally:
        trainer.shutdown()

if __name__ == "__main__":
    parse_args_with_extra_kwargs = click.option("--trainer-id", type=int)(parse_args)
    parse_args_with_extra_kwargs = click.option("--run-locally", is_flag=True)(parse_args_with_extra_kwargs)
    parse_args_with_extra_kwargs = click.option("--disable-quant", is_flag=True)(parse_args_with_extra_kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*is used more than once. Remove its duplicate as parameters should be unique.*")
        res = parse_args_with_extra_kwargs(standalone_mode=False)
        if isinstance(res, int):
            quit() # Help has been called
        elif isinstance(res, tuple):
            cfg, extra_kwargs = res
            main(cfg, **extra_kwargs)
        else:
            raise ValueError(f"Unexpected return type: {type(res)}")
