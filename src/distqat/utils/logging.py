import wandb
import logging
from pathlib import Path
from typing import Optional

from hivemind.utils.logging import get_logger
from hivemind.dht import DHT
from hivemind.utils import get_dht_time, DHTExpiration
logger = get_logger(__name__)


def setup_file_logging(log_file_path: Path, wandb_enabled: bool = False, disable_console: bool = True):
    """
    Set up file logging for the root logger.
    
    Args:
        log_file_path: Path to the log file
        wandb_enabled: Whether to track the log file with wandb
        disable_console: If True, remove console/stream handlers to log only to file
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger (hivemind uses root logger)
    root_logger = logging.getLogger()
    
    # Remove console handlers
    if disable_console:
        handlers_to_remove = [
            handler for handler in root_logger.handlers
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        ]
        for handler in handlers_to_remove:
            root_logger.removeHandler(handler)
    
    existing_file_handler = None
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file_path.absolute()):
            existing_file_handler = handler
            break
    
    if existing_file_handler is None:
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        root_logger.addHandler(file_handler)
    
    track_log_file(log_file_path, wandb_enabled=wandb_enabled, live=True)


def track_log_file(path: Path, wandb_enabled: bool = True, live: bool = True):
    if not wandb_enabled:
        return
    try:
        policy = "live" if live else "now"
        wandb.save(str(path), policy=policy)
    except Exception as e:
        logger.warning(f"Failed to register log file '{path}' with wandb: {e}")


def store_wandb_run_id(
    dht: DHT, experiment_prefix: str, wandb_run_id: str, expiration: DHTExpiration = 3600, wait: bool = True
) -> bool:
    """
    Store wandb_run_id in the DHT for sharing across processes.
    
    :param dht: DHT instance to store run ID in
    :param experiment_prefix: experiment prefix to use as key namespace
    :param wandb_run_id: wandb run ID to store
    :param expiration: expiration time in seconds (default: 1 hour)
    :param wait: if True, awaits for storage to finish
    :returns: True if storage succeeded, False otherwise
    """
    key = f"{experiment_prefix}_wandb_run_id"
    expiration_time = get_dht_time() + expiration
    return dht.store(key, wandb_run_id, expiration_time=expiration_time, return_future=not wait)


def get_wandb_run_id(
    dht: DHT, experiment_prefix: str, latest: bool = True
) -> Optional[str]:
    """
    Retrieve wandb_run_id from the DHT.
    
    :param dht: DHT instance to query
    :param experiment_prefix: experiment prefix used as key namespace
    :param latest: if True, get the latest value
    :returns: wandb_run_id if found, None otherwise
    """
    key = f"{experiment_prefix}_wandb_run_id"
    result = dht.get(key, latest=latest)
    if result is not None and result.value is not None:
        return str(result.value)
    return None
