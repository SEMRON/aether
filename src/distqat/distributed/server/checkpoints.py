import os
import threading
from datetime import datetime
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Dict

import torch

from hivemind.utils.logging import get_logger

from distqat.distributed.server.expert_backend import ExpertBackend

logger = get_logger(__name__)


def is_directory(directory: Path):
    """
    Validate that the given path is a valid directory.
    
    :param directory: path to validate
    :returns: True if path is a valid directory
    :raises: AssertionError if path is None, doesn't exist, or is not a directory
    """
    assert directory is not None
    assert directory.exists()
    assert directory.is_dir()
    return True


def copy_tree(src: str, dst: str):
    """
    Recursively copy a directory tree from source to destination.
    
    Creates the destination directory if it doesn't exist and copies all files
    and subdirectories from src to dst, preserving file metadata.
    
    :param src: source directory path
    :param dst: destination directory path
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        src_entry = os.path.join(src, item)
        dst_entry = os.path.join(dst, item)
        if os.path.isdir(src_entry):
            copy_tree(src_entry, dst_entry)
        else:
            copy2(src_entry, dst_entry)


class CheckpointSaver(threading.Thread):
    """
    A background thread that periodically saves expert checkpoints to disk.
    
    This thread runs continuously and creates timestamped checkpoints of all expert backends,
    allowing for recovery and persistence across server restarts.
    
    :param expert_backends: dictionary of expert backends to checkpoint
    :param checkpoint_dir: directory where checkpoints will be saved
    :param update_period: how often (in seconds) to save checkpoints
    """
    def __init__(self, expert_backends: Dict[str, ExpertBackend], checkpoint_dir: Path, update_period: int):
        super().__init__()
        assert is_directory(checkpoint_dir)
        self.expert_backends = expert_backends
        self.update_period = update_period
        self.checkpoint_dir = checkpoint_dir
        self.stop = threading.Event()

        # create expert directories to ensure that the directory is writable and checkpoints can be loaded
        store_experts(self.expert_backends, self.checkpoint_dir)

    def run(self) -> None:
        """
        Main thread loop that periodically saves expert checkpoints.
        
        Continuously creates timestamped checkpoints of all experts at regular
        intervals until the thread is signaled to stop.
        """
        while not self.stop.wait(self.update_period):
            store_experts(self.expert_backends, self.checkpoint_dir)


def store_experts(experts: Dict[str, ExpertBackend], checkpoint_dir: Path):
    """
    Save checkpoints for all experts to the specified directory.
    
    Creates timestamped checkpoint files for each expert and updates symlinks
    to point to the latest checkpoint for each expert.
    
    :param experts: dictionary mapping expert names to ExpertBackend instances
    :param checkpoint_dir: directory where checkpoints will be saved
    """
    logger.debug(f"Storing experts at {checkpoint_dir.absolute()}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    assert is_directory(checkpoint_dir)
    timestamp = datetime.now().isoformat(sep="_")
    with TemporaryDirectory() as tmpdirname:
        for expert_name, expert_backend in experts.items():
            expert_dir = Path(tmpdirname) / expert_name
            expert_dir.mkdir()
            checkpoint_name = expert_dir / f"checkpoint_{timestamp}.pt"
            torch.save(expert_backend.get_full_state(), checkpoint_name)
            os.symlink(checkpoint_name, expert_dir / "checkpoint_last.pt")
        copy_tree(tmpdirname, str(checkpoint_dir))


def load_experts(experts: Dict[str, ExpertBackend], checkpoint_dir: Path):
    """
    Load the latest checkpoints for all experts from the specified directory.
    
    Attempts to restore each expert's state from the most recent checkpoint file.
    Logs warnings for experts that don't have available checkpoints.
    
    :param experts: dictionary mapping expert names to ExpertBackend instances
    :param checkpoint_dir: directory where checkpoints are stored
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    assert is_directory(checkpoint_dir)
    for expert_name, expert in experts.items():
        checkpoints_folder = checkpoint_dir / expert_name
        latest_checkpoint = checkpoints_folder / "checkpoint_last.pt"
        if latest_checkpoint.exists():
            expert.load_full_state(torch.load(latest_checkpoint))
        else:
            logger.warning(f"Failed to load checkpoint for expert {expert_name}")
