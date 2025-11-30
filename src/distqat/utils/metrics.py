from pydantic import BaseModel, conint, StrictFloat
from hivemind.dht import DHT
from hivemind.utils import get_dht_time
from hivemind.utils.logging import get_logger
import torch

logger = get_logger(__name__)

class LocalMetrics(BaseModel):
    step: conint(ge=0, strict=True)
    trainer_id: conint(ge=-1, strict=True)
    loss: StrictFloat
    sps: StrictFloat


class MetricsLogger:
    """
    This callback logs metrics to the DHT for the monitor to collect.
    No Wandb initialization here - that's handled centrally by the monitor.
    """

    def __init__(
            self,
            dht: DHT,
            model: torch.nn.Module,
            local_public_key: bytes,
            experiment_prefix: str,
            statistics_expiration: float,
            trainer_id: int,
        ):
            super().__init__()
            self.model = model
            self.dht = dht
            self.local_public_key = local_public_key
            self.experiment_prefix = experiment_prefix
            self.statistics_expiration = statistics_expiration
            self.last_reported_collaboration_step = -1
            self.trainer_id = trainer_id
            logger.debug(f"CollaborativeCallback initialized")

    def on_step_end(self, global_step, loss, sps=0.0):
        if global_step != self.last_reported_collaboration_step:
            self.last_reported_collaboration_step = global_step

            try:
                statistics = LocalMetrics(
                    step=global_step,
                    trainer_id=self.trainer_id,
                    loss=loss,
                    sps=sps,
                )

                self.dht.store(
                    key=self.experiment_prefix + "_metrics",
                    subkey=self.local_public_key,
                    value=statistics.dict(),
                    expiration_time=get_dht_time() + self.statistics_expiration,
                    return_future=True,
                )
                logger.debug(f"Metrics stored to DHT: {statistics}")
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                logger.warning(f"Failed to store metrics to DHT: {e}")