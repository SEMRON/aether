from socket import gethostname
from typing import Dict, Optional, Any
from pydantic.v1 import BaseModel, conint, StrictFloat
import torch
from hivemind.utils import get_logger, get_dht_time
from hivemind.utils.logging import use_hivemind_log_handler
from hivemind.dht import DHT
from contextlib import nullcontext


from distqat.config import Config, ModelPipelineConfig
from distqat.distributed.client import BalancedRemoteExpert
from distqat.distributed.client.local_backend_expert import LocalBackendExpert
from distqat.distributed.optim.collaborative import CollaborativeOptimizer
from distqat.models import get_model
from distqat.optimizers import get_diloco_optimizer_cls_kwargs
from distqat.attach import attach_quantizers
from distqat.utils.metrics import MetricsLogger, LocalMetrics
from distqat.utils.baseline_checkpoints import BaselineCheckpointSaver, load_checkpoint
from distqat.utils.compression import get_compression_kwargs

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


class RemoteModel(torch.nn.Module):
    """
    A model that executes pipeline stages on remote experts via hivemind DHT.
    
    Args:
        pipeline_id: Unique identifier for this pipeline instance
        dht: DHT instance for discovering and communicating with remote experts
        cfg: Configuration defining the pipeline stages and timeouts
    """
    def __init__(self, pipeline_id, dht, cfg: ModelPipelineConfig, *, local_expert_backends: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.model_pipeline = torch.nn.ModuleList()
        for pipeline_step_cfg in cfg.pipeline:
            _, stage = pipeline_step_cfg.model_name.split(".")
            # Full uid format in this codebase is: "{stage}.0.{pipeline_id}.0"
            # RemoteModel uses uid_prefix "{stage}.0.{pipeline_id}." to discover via DHT.
            uid = f"{stage}.0.{pipeline_id}.0"
            if local_expert_backends is not None and uid in local_expert_backends:
                pipeline_step = LocalBackendExpert(backend=local_expert_backends[uid])
            else:
                pipeline_step = BalancedRemoteExpert(
                    dht=dht,
                    forward_timeout=cfg.forward_timeout,
                    backward_timeout=cfg.backward_timeout,
                    uid_prefix=f"{stage}.0.{pipeline_id}.",
                    initial_throughput=0.01,
                )
            self.model_pipeline.append(pipeline_step)

    def forward(self, x, labels=None):
        num_stages = len(self.model_pipeline)
        # Handle both tuples and lists (pin_memory=True converts tuples to lists)
        is_sequence = isinstance(x, (tuple, list))
        for idx, pipeline_step in enumerate(self.model_pipeline):
            is_last_stage = (idx == num_stages - 1)
            if is_sequence:
                if labels is not None and is_last_stage:
                    x = pipeline_step(*x, labels)
                else:
                    x = pipeline_step(*x)
                # After first stage, output is typically a single tensor, update is_sequence flag
                is_sequence = isinstance(x, (tuple, list))
            else:
                if labels is not None and is_last_stage:
                    x = pipeline_step(x, labels)
                else:
                    x = pipeline_step(x)
        return x

    def forward_averaged(self, x, labels=None):
        """
        Forward pass using globally averaged model weights on all pipeline stages.
        
        This is useful for distributed RL where rollouts should be collected under
        a consistent policy (the averaged model) across all workers.
        
        Note: This is inference-only (no gradients).
        """
        num_stages = len(self.model_pipeline)
        is_sequence = isinstance(x, (tuple, list))
        for idx, pipeline_step in enumerate(self.model_pipeline):
            is_last_stage = (idx == num_stages - 1)
            # Use forward_averaged if available, otherwise fall back to regular forward
            forward_fn = getattr(pipeline_step, 'forward_averaged', pipeline_step.forward)
            if is_sequence:
                if labels is not None and is_last_stage:
                    x = forward_fn(*x, labels)
                else:
                    x = forward_fn(*x)
                is_sequence = isinstance(x, (tuple, list))
            else:
                if labels is not None and is_last_stage:
                    x = forward_fn(x, labels)
                else:
                    x = forward_fn(x)
        return x

    def shutdown(self):
        for pipeline_step in self.model_pipeline:
            if hasattr(pipeline_step, 'expert_balancer'):
                pipeline_step.expert_balancer.shutdown()

    def parameters(self):
        raise NotImplementedError("RemoteModel does not have parameters")

    def evaluate(self, step):
        """Not implemented for RemoteModel"""
        pass

class BaselineModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()  
        self.model_pipeline = torch.nn.ModuleList()
        for pipeline_step_cfg in config.model_pipeline.pipeline:
            pipeline_step = get_model(config, pipeline_step_cfg)
            self.model_pipeline.append(pipeline_step)

    def forward(self, x, labels=None):
        num_stages = len(self.model_pipeline)
        # Handle both tuples and lists (pin_memory=True converts tuples to lists)
        is_sequence = isinstance(x, (tuple, list))
        for idx, pipeline_step in enumerate(self.model_pipeline):
            is_last_stage = (idx == num_stages - 1)
            if is_sequence:
                if labels is not None and is_last_stage:
                    x = pipeline_step(*x, labels)
                else:
                    x = pipeline_step(*x)
                # After first stage, output is typically a single tensor, update is_sequence flag
                is_sequence = isinstance(x, (tuple, list))
            else:
                if labels is not None and is_last_stage:
                    x = pipeline_step(x, labels)
                else:
                    x = pipeline_step(x)
        return x
    
    def forward_averaged(self, x, labels=None):
        """
        Forward pass using globally averaged model weights on all pipeline stages.
        
        This is useful for distributed RL where rollouts should be collected under
        a consistent policy (the averaged model) across all workers.
        
        Note: This is inference-only (no gradients).
        """
        num_stages = len(self.model_pipeline)
        is_sequence = isinstance(x, (tuple, list))
        for idx, pipeline_step in enumerate(self.model_pipeline):
            is_last_stage = (idx == num_stages - 1)
            # Use forward_averaged if available, otherwise fall back to regular forward
            forward_fn = getattr(pipeline_step, 'forward_averaged', pipeline_step.forward)
            if is_sequence:
                if labels is not None and is_last_stage:
                    x = forward_fn(*x, labels)
                else:
                    x = forward_fn(*x)
                is_sequence = isinstance(x, (tuple, list))
            else:
                if labels is not None and is_last_stage:
                    x = forward_fn(x, labels)
                else:
                    x = forward_fn(x)
        return x

    def evaluate(self, step):
        """Evaluate the model every eval_every steps"""
        for pipeline_step in self.model_pipeline:
            if hasattr(pipeline_step, 'evaluate'):
                pipeline_step.evaluate(step)

class SwarmModel(torch.nn.Module):
    """
    A distributed model that connects to hivemind swarm for collaborative training.
    
    This model uses remote experts for computation and includes collaborative training
    callbacks for metrics reporting and wandb integration.
    
    Args:
        config: Full configuration including network, model pipeline, and experiment settings
        trainer_id: Unique identifier for this trainer instance
    """
    def __init__(
        self,
        config: Config,
        trainer_id: int,
        *,
        dht: Optional[DHT] = None,
        local_expert_backends: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        initial_peers = config.network.initial_peers
        host_maddrs = config.network.host_maddrs
        announce_maddrs = config.network.announce_maddrs
        statistics_expiration = config.diloco.metadata_expiration

        self._owns_dht = dht is None
        self.dht = dht or DHT(
            start=True,
            initial_peers=initial_peers,
            client_mode=config.network.client_mode,
            host_maddrs=host_maddrs,
            announce_maddrs=announce_maddrs,
        )

        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {config.network.initial_peers}")
        logger.info(f"Host maddrs = {host_maddrs}, announce maddrs = {announce_maddrs}")

        hostname = gethostname()
        import base64, os
        random_b64 = base64.b64encode(os.urandom(6)).decode("utf-8")[:8]

        self.model = RemoteModel(
            trainer_id,
            self.dht,
            config.model_pipeline,
            local_expert_backends=local_expert_backends,
        )

        self.metrics_logger = MetricsLogger(
            self.dht,
            self.model,
            hostname + "." + random_b64,
            config.experiment_prefix,
            statistics_expiration,
            trainer_id,
        )

    def shutdown(self):
        print("SHUTDOWN Model")

        try:
            if hasattr(self, "metrics_logger") and hasattr(self.metrics_logger, "shutdown"):
                self.metrics_logger.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down metrics logger: {e}")

        try:
            self.model.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down model: {e}")
        if self._owns_dht:
            print("SHUTDOWN DHT")
            try:
                self.dht.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down DHT: {e}")

    def parameters(self):
        yield from self.model.parameters()

    def forward(self, x, labels=None):
        return self.model(x, labels)

    def forward_averaged(self, x, labels=None):
        """
        Forward pass using globally averaged model weights.
        
        Useful for distributed RL where rollouts should be collected under
        a consistent policy across all workers.
        """
        return self.model.forward_averaged(x, labels)
    
    def evaluate(self, step):
        pass
    
    def post_optimizer_callback(self, global_step, loss, **_kwargs):
        self.metrics_logger.on_step_end(global_step, loss)

    
class SwarmBaselineModel(torch.nn.Module):
    def __init__(self, 
        config: Config,
        trainer_id: int,
        disable_quant: bool,
    ):
        super().__init__()


        config = config.model_copy()
        self.config = config
        self.dht = DHT(
            start=True,
            initial_peers=config.network.initial_peers,
            client_mode=False,
            host_maddrs=config.network.host_maddrs,
            announce_maddrs=config.network.announce_maddrs,
        )


        hostname = gethostname()
        import base64, os
        random_b64 = base64.b64encode(os.urandom(6)).decode("utf-8")[:8]
        statistics_expiration = config.diloco.metadata_expiration

        self.device = config.device

        model = BaselineModel(config=config)
        if not disable_quant:
            model, avg_only_params = attach_quantizers(model, config.quant)
        else:
            avg_only_params = []
        model.to(config.device)

        self.model = model
        
        run_id = f"{config.experiment_prefix}_baseline"

        # DataServer model uses trainer_id = -2 and needs to use run_id = "0" to average with other swarm models
        # Currently only used by PPO and would only work with full stage models
        if trainer_id == -2:
            run_id = f"{config.experiment_prefix}_0"
        is_biggan = config.model_pipeline.pipeline[0].model_name == "biggan.full"
        expert = model.model_pipeline[0] if is_biggan else None

        compression = get_compression_kwargs(config.network.hivemind_compression)

        # Create a modified diloco config for the optimizer
        diloco_config = config.diloco
        # Make the outer optimizer a no-op by setting lr=1, momentum=0, nesterov=false.
        # This preserves the DiLoCo infrastructure (scheduler, progress tracking, DHT)
        # but the outer step just preserves the inner training result: W_new.
        # Math: param = W_old - lr * (W_old - W_new) = W_old - 1*(W_old - W_new) = W_new
        diloco_config = diloco_config.model_copy(deep=True)
        diloco_config.outer_optim.sgd_lr = 1.0
        diloco_config.outer_optim.sgd_momentum = 0.0
        diloco_config.outer_optim.sgd_nesterov = False
        logger.info(
            "outer optimizer set to no-op (lr=1, momentum=0, nesterov=false). "
            "Inner training will be fully preserved at each outer step."
        )

        optimizer_cls, optimizer_kwargs = get_diloco_optimizer_cls_kwargs(
            run_id=run_id,
            config=diloco_config,
            compression=compression,
        )
        # Baseline trainer performs gradient accumulation explicitly in `SwarmTrainer.step()`.
        # If we also configure DiLoCoOptimizer.gradient_accumulation_steps > 1, we effectively "double accumulate":
        # the trainer accumulates gradients over N microbatches, but DiLoCoOptimizer will only step every N calls.
        #
        # For baseline, keep DiLoCo's internal accumulation at 1 and treat batch_size_per_step as the *effective*
        # batch per optimizer update so progress/speed accounting stay meaningful.
        self._effective_batch_size_per_step = int(diloco_config.batch_size_per_step) * int(
            diloco_config.gradient_accumulation_steps
        )
        if int(diloco_config.gradient_accumulation_steps) > 1:
            logger.warning(
                "Baseline trainer uses explicit gradient accumulation "
                f"(trainer grad_accum={diloco_config.gradient_accumulation_steps}). "
                "Overriding DiLoCoOptimizer.gradient_accumulation_steps to 1 and setting "
                f"DiLoCoOptimizer.batch_size_per_step={self._effective_batch_size_per_step} "
                "to avoid double-accumulation and scheduler/step mismatches."
            )
            optimizer_kwargs["gradient_accumulation_steps"] = 1
            optimizer_kwargs["batch_size_per_step"] = self._effective_batch_size_per_step
        self.optimizer = optimizer_cls(
            params=model.parameters(),
            avg_only_params=avg_only_params,
            expert=expert,
            dht=self.dht,
            **optimizer_kwargs,
        )

        self.update_count = 0
        self.examples_processed = 0

        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_keep_history = config.checkpoint_keep_history

        if self.checkpoint_dir is not None:
            # Load full state (model weights + optimizer + stats) when available.
            load_checkpoint(self, self.checkpoint_dir / "baseline")

            self.checkpoint_saver = BaselineCheckpointSaver(
                model=self,
                checkpoint_dir=self.checkpoint_dir / "baseline",
                update_period=config.checkpoint_update_period,
                keep_history=self.checkpoint_keep_history,
            )
            self.checkpoint_saver.start()
        else:
            self.checkpoint_saver = None

        self.metrics_logger = MetricsLogger(
            self.dht,
            self.model,
            hostname + "." + random_b64,
            config.experiment_prefix,
            statistics_expiration,
            trainer_id,
            # Data server (trainer_id == -2) emits PPO episodic metrics based on env steps.
            # Keep these separate from trainer loss metrics so the monitor can aggregate both cleanly.
            key_suffix="_rl_metrics" if trainer_id == -2 else "_metrics",
        )
        

    def shutdown(self):
        try:
            if hasattr(self, "metrics_logger") and hasattr(self.metrics_logger, "shutdown"):
                self.metrics_logger.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down metrics logger: {e}")

        try:
            if self.checkpoint_saver is not None:
                self.checkpoint_saver.stop.set()
                self.checkpoint_saver.join(timeout=10)
        except Exception as e:
            logger.warning(f"Error shutting down checkpoint saver: {e}")

        try:
            if hasattr(self.optimizer, "shutdown") and callable(getattr(self.optimizer, "shutdown")):
                self.optimizer.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down optimizer: {e}")
        
        try:
            self.dht.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down DHT: {e}")

    def forward(self, x, labels=None):
        autocast_dtype = None
        if self.config.data.precision in ("fp16-mixed", "bf16-mixed"):
            autocast_dtype = torch.bfloat16 if self.config.data.precision == "bf16-mixed" else torch.float16
            
        with torch.amp.autocast(device_type=self.device, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext():
            if isinstance(x, tuple) or isinstance(x, list):
                x = tuple(item.to(self.device) if hasattr(item, "to") else item for item in x)
                y = self.model(x, labels)
            else:
                x = x.to(self.device)
                y = self.model(x, labels)
            return y
    
    def forward_averaged(self, x, labels=None):
        """
        Forward pass using globally averaged model weights.
        
        Useful for distributed RL where rollouts should be collected under
        a consistent policy across all workers.
        """
        return self.model.forward_averaged(x, labels)
    
    def evaluate(self, step):
        """Evaluate the model every eval_every steps"""
        self.model.evaluate(step)

    def grad_accum_step_callback(self, grad_accum_step, global_step, loss):
        logger.info(f"Grad accum step callback: {grad_accum_step}, {global_step}, {loss}")
        self.metrics_logger.on_step_end(global_step, loss)

    def post_optimizer_callback(self, global_step, loss, **_kwargs):
        self.update_count += 1
        eff_bs = getattr(
            self,
            "_effective_batch_size_per_step",
            int(self.config.diloco.batch_size_per_step) * int(self.config.diloco.gradient_accumulation_steps),
        )
        self.examples_processed += int(eff_bs)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.metrics_logger.on_step_end(global_step, loss)

    def get_stats(self):
        return {"updates": self.update_count, "examples_processed": self.examples_processed}
    
    def get_full_state(self):
        """
        Return the current state of the baseline model (including batch processing statistics)
        """
        full_state = {
            "stats": self.get_stats(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return full_state

    def load_full_state(self, state_dict: Dict):
        """
        Restore a baseline checkpoint produced by get_full_state().
        This is best-effort for optimizer state (may fail if param groups changed).
        """
        # Accept both formats:
        # - extended: {"stats":..., "model":..., "optimizer":...}
        # - legacy: raw model.state_dict()
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unexpected checkpoint payload type: {type(state_dict)}")

        if "model" in state_dict:
            model_state = state_dict.get("model", {})
            try:
                self.model.load_state_dict(model_state, strict=False)
            except Exception as e:
                logger.warning(f"Failed to restore model weights from checkpoint: {e}")

            stats = state_dict.get("stats", {}) or {}
            try:
                self.update_count = int(stats.get("updates", self.update_count))
            except Exception:
                pass
            try:
                self.examples_processed = int(stats.get("examples_processed", self.examples_processed))
            except Exception:
                pass

            opt_state = state_dict.get("optimizer", None)
            if opt_state is not None:
                try:
                    self.optimizer.load_state_dict(opt_state)
                except Exception as e:
                    logger.warning(f"Failed to restore optimizer state from checkpoint (continuing): {e}")
        else:
            # Legacy payload: treat as model weights only.
            self.model.load_state_dict(state_dict, strict=False)