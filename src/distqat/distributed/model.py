from socket import gethostname

from pydantic.v1 import BaseModel, conint, StrictFloat
import torch
from hivemind.utils import get_logger, get_dht_time
from hivemind.utils.logging import use_hivemind_log_handler
from hivemind.dht import DHT

from distqat.config import Config, ModelPipelineConfig
from distqat.distributed.client import BalancedRemoteExpert
from distqat.distributed.optim.collaborative import CollaborativeOptimizer
from distqat.models import get_model
from distqat.optimizers import get_diloco_optimizer_cls_kwargs
from distqat.attach import attach_quantizers
from distqat.utils.metrics import MetricsLogger, LocalMetrics
from distqat.utils.baseline_checkpoints import BaselineCheckpointSaver, load_checkpoint

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
    def __init__(self, pipeline_id, dht, cfg: ModelPipelineConfig):
        super().__init__()

        self.model_pipeline = torch.nn.ModuleList()
        for pipeline_step_cfg in cfg.pipeline:
            _, stage = pipeline_step_cfg.model_name.split(".")
            pipeline_step = BalancedRemoteExpert(
                dht=dht,
                forward_timeout=cfg.forward_timeout,
                backward_timeout=cfg.backward_timeout,
                uid_prefix=f"{stage}.0.{pipeline_id}.",
                initial_throughput=0.01,
            )
            self.model_pipeline.append(pipeline_step)

    def forward(self, x):
        for pipeline_step in self.model_pipeline:
            if isinstance(x, tuple):
                x = pipeline_step(*x)
            else:
                x = pipeline_step(x)
        return x

    def shutdown(self):
        for pipeline_step in self.model_pipeline:
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

    def forward(self, x):
        for pipeline_step in self.model_pipeline:
            if isinstance(x, tuple):
                x = pipeline_step(*x)
            else:
                x = pipeline_step(x)
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
    ):
        super().__init__()

        initial_peers = config.network.initial_peers
        host_maddrs = config.network.host_maddrs
        announce_maddrs = config.network.announce_maddrs
        statistics_expiration = 300

        self.dht = DHT(
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

        self.model = RemoteModel(trainer_id, self.dht, config.model_pipeline)

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
            self.model.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down model: {e}")
        print("SHUTDOWN DHT")

        try:
            self.dht.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down DHT: {e}")

    def parameters(self):
        yield from self.model.parameters()

    def forward(self, x):
        return self.model(x)
    
    def evaluate(self, step):
        pass
    
    def post_optimizer_callback(self, global_step, loss):
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
        statistics_expiration = 300

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
        optimizer_cls, optimizer_kwargs = get_diloco_optimizer_cls_kwargs(
            run_id=run_id, 
            config=config.diloco,
        )
        self.optimizer = optimizer_cls(
            params=model.parameters(),
            avg_only_params=avg_only_params,
            expert=model.model_pipeline[0] if config.model_pipeline.pipeline[0].model_name == "biggan.full" else None,
            dht=self.dht,
            **optimizer_kwargs,
        )

        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_keep_history = config.checkpoint_keep_history

        if self.checkpoint_dir is not None:
            load_checkpoint(self.model, self.checkpoint_dir / "baseline")

            self.checkpoint_saver = BaselineCheckpointSaver(
                model=self.model,
                checkpoint_dir=self.checkpoint_dir / "baseline",
                update_period=10,
                keep_history=self.checkpoint_keep_history
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
            if isinstance(self.optimizer, CollaborativeOptimizer):
                if self.checkpoint_saver is not None:
                    self.checkpoint_saver.stop.set()
                    self.checkpoint_saver.join()
                self.optimizer.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down optimizer: {e}")
        
        try:
            self.dht.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down DHT: {e}")

    def forward(self, x):
        if isinstance(x, tuple):
            x = tuple(item.to(self.device) if hasattr(item, "to") else item for item in x)
            y = self.model(x)
        else:
            x = x.to(self.device)
            y = self.model(x)
        if isinstance(y, tuple):
            y = tuple(item.cpu() for item in y)
        else:
            y = y.cpu()
        return y

    def evaluate(self, step):
        """Evaluate the model every eval_every steps"""
        self.model.evaluate(step)

    def post_optimizer_callback(self, global_step, loss):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.metrics_logger.on_step_end(global_step, loss)
