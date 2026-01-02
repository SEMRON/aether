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
    :param keep_history: if True, save timestamped checkpoints and update checkpoint_last.pt;
        if False, only keep checkpoint_last.pt and overwrite it each time
    """
    def __init__(
        self,
        expert_backends: Dict[str, ExpertBackend],
        checkpoint_dir: Path,
        update_period: int,
        *,
        keep_history: bool = True,
    ):
        super().__init__()
        assert is_directory(checkpoint_dir)
        self.expert_backends = expert_backends
        self.update_period = update_period
        self.checkpoint_dir = checkpoint_dir
        self.keep_history = keep_history
        self.stop = threading.Event()

        # create expert directories to ensure that the directory is writable and checkpoints can be loaded
        store_experts(self.expert_backends, self.checkpoint_dir, keep_history=self.keep_history)

    def run(self) -> None:
        """
        Main thread loop that periodically saves expert checkpoints.
        
        Continuously creates timestamped checkpoints of all experts at regular
        intervals until the thread is signaled to stop.
        """
        while not self.stop.wait(self.update_period):
            store_experts(self.expert_backends, self.checkpoint_dir, keep_history=self.keep_history)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def store_experts(experts: Dict[str, ExpertBackend], checkpoint_dir: Path, *, keep_history: bool = True):
    """
    Save checkpoints for all experts to the specified directory.
    
    If keep_history is True, creates timestamped checkpoint files for each expert and
    updates checkpoint_last.pt to point to the latest checkpoint for each expert.
    If keep_history is False, overwrites checkpoint_last.pt and removes any
    timestamped checkpoint_*.pt files for each expert.
    
    :param experts: dictionary mapping expert names to ExpertBackend instances
    :param checkpoint_dir: directory where checkpoints will be saved
    :param keep_history: whether to save timestamped history or only keep the latest checkpoint
    """
    logger.debug(f"Storing experts at {checkpoint_dir.absolute()}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    assert is_directory(checkpoint_dir)
    timestamp = datetime.now().isoformat(sep="_")

    for expert_name, expert_backend in experts.items():
        expert_dir = checkpoint_dir / expert_name
        expert_dir.mkdir(parents=True, exist_ok=True)

        if keep_history:
            checkpoint_path = expert_dir / f"checkpoint_{timestamp}.pt"
            tmp_checkpoint_path = expert_dir / f".checkpoint_{timestamp}.pt.tmp"
            torch.save(expert_backend.get_full_state(), tmp_checkpoint_path)
            os.replace(tmp_checkpoint_path, checkpoint_path)

            # Update checkpoint_last.pt as a relative symlink for portability.
            # Use a temp symlink + replace for atomic update.
            last_path = expert_dir / "checkpoint_last.pt"
            tmp_last_path = expert_dir / ".checkpoint_last.pt.tmp"
            _safe_unlink(tmp_last_path)
            os.symlink(checkpoint_path.name, tmp_last_path)
            os.replace(tmp_last_path, last_path)
        else:
            # Only keep checkpoint_last.pt, overwriting it each time.
            last_path = expert_dir / "checkpoint_last.pt"
            tmp_last_path = expert_dir / ".checkpoint_last.pt.tmp"
            torch.save(expert_backend.get_full_state(), tmp_last_path)
            os.replace(tmp_last_path, last_path)

            # Remove any historical timestamped checkpoints to enforce "only last".
            for old_ckpt in expert_dir.glob("checkpoint_*.pt"):
                # NOTE: "checkpoint_last.pt" matches "checkpoint_*.pt", so we must not delete it.
                if old_ckpt.name == "checkpoint_last.pt":
                    continue
                _safe_unlink(old_ckpt)


def load_experts(experts: Dict[str, ExpertBackend], checkpoint_dir: Path):
    """
    Load the latest checkpoints for all experts from the specified directory.
    
    Attempts to restore each expert's state from the most recent checkpoint file.
    Falls back to baseline checkpoint if expert-specific checkpoint doesn't exist.
    Logs warnings for experts that don't have available checkpoints.
    
    :param experts: dictionary mapping expert names to ExpertBackend instances
    :param checkpoint_dir: directory where checkpoints are stored
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    assert is_directory(checkpoint_dir)
    
    # Check for baseline checkpoint as fallback
    baseline_checkpoint = checkpoint_dir / "baseline" / "checkpoint_last.pt"
    baseline_state = None
    if baseline_checkpoint.exists():
        logger.info(f"Found baseline checkpoint at {baseline_checkpoint}, will use as fallback")
        baseline_state = torch.load(baseline_checkpoint)
    
    for expert_name, expert in experts.items():
        checkpoints_folder = checkpoint_dir / expert_name
        latest_checkpoint = checkpoints_folder / "checkpoint_last.pt"
        if latest_checkpoint.exists():
            expert.load_full_state(torch.load(latest_checkpoint))
        elif baseline_state is not None:
            logger.info(f"Using baseline checkpoint for expert {expert_name}")
            expert.load_full_state(baseline_state)
        else:
            logger.warning(f"Failed to load checkpoint for expert {expert_name}")
