import warnings
import os
import click
import queue, threading, time


if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ and os.environ["HF_HUB_ENABLE_HF_TRANSFER"] != "0":
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import signal

from multiprocessing.managers import BaseManager
from multiprocessing import shared_memory
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset, get_worker_info
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from distqat.config import parse_args, Config
from distqat.data import get_train_val_datasets, collate_fn
from distqat.utils.hash import hash64, _to_bytes

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

# ---------- Buffered shuffle iterable ----------

class BufferedShuffleIterable(IterableDataset):
    """
    Wraps an iterable of (uid, sample) and emits the same stream with
    an online shuffle buffer. Worker-safe: shards items across workers by uid.
    """
    def __init__(self, src, buffer_size: int = 8192, seed: int = 0):
        self.src = src
        self.buffer_size = max(2, buffer_size)
        self.seed = seed

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nworkers = wi.num_workers if wi else 1

        rng = torch.Generator().manual_seed(self.seed + wid)
        buf = []

        it = iter(self.src)

        # fill
        while len(buf) < self.buffer_size:
            try:
                uid, sample = next(it)
                # shard by worker to avoid duplicates
                if (hash64(_to_bytes(uid)) % nworkers) != wid:
                    continue
                buf.append((uid, sample))
            except StopIteration:
                break

        # stream
        while buf:
            idx = int(torch.randint(len(buf), (1,), generator=rng).item())
            uid, sample = buf.pop(idx)
            yield uid, sample
            # replenish
            while True:
                try:
                    u2, s2 = next(it)
                    if (hash64(_to_bytes(u2)) % nworkers) != wid:
                        continue
                    buf.append((u2, s2))
                    break
                except StopIteration:
                    # source exhausted (finite); we’ll drain buffer
                    break

# ---------- SHM pool ----------

class SharedBatchPool:
    """Fixed-size pool of SHM slots: inputs, labels (extend if needed)."""
    def __init__(self, pool_size=32):
        self.pool_size = pool_size
        self.slots = [None] * pool_size
        self.free = queue.Queue()
        for i in range(pool_size):
            self.free.put(i)

    def _ensure(self, slot_id, key, nbytes):
        slot = self.slots[slot_id] or {"inputs": None, "labels": None}
        shm = slot[key]
        if shm is None or shm.size < nbytes:
            if shm is not None:
                try: shm.close(); shm.unlink()
                except: pass
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            slot[key] = shm
            self.slots[slot_id] = slot
        return shm

    def put_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        slot_id = self.free.get()  # blocks if none free
        x = inputs.detach().contiguous().cpu()
        y = labels.detach().contiguous().cpu()

        x_nbytes = x.element_size() * x.nelement()
        y_nbytes = y.element_size() * y.nelement()

        x_shm = self._ensure(slot_id, "inputs", x_nbytes)
        y_shm = self._ensure(slot_id, "labels", y_nbytes)

        x_bytes = x.numpy().view(np.uint8).reshape(-1)
        y_bytes = y.numpy().view(np.uint8).reshape(-1)
        np.frombuffer(x_shm.buf, dtype=np.uint8, count=x_nbytes)[:] = x_bytes
        np.frombuffer(y_shm.buf, dtype=np.uint8, count=y_nbytes)[:] = y_bytes

        return {
            "slot": slot_id,
            "inputs": {"name": x_shm.name, "shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", "")},
            "labels": {"name": y_shm.name, "shape": list(y.shape), "dtype": str(y.dtype).replace("torch.", "")},
        }

    def release(self, slot_id: int):
        self.free.put(slot_id)

    def shutdown(self):
        for s in self.slots:
            if not s: continue
            for k in ("inputs", "labels"):
                shm = s[k]
                if shm:
                    try: shm.close(); shm.unlink()
                    except: pass

# ---------- Manager (IPC queues) ----------

class ServerManager(BaseManager): pass

def start_manager(work_q, free_q, address=("127.0.0.1", 52555), authkey=b"distqat"):
    ServerManager.register("get_work_q", callable=lambda: work_q)
    ServerManager.register("get_free_q", callable=lambda: free_q)
    mgr = ServerManager(address=address, authkey=authkey)
    mgr.start()
    logger.info(f"Data server manager started at {address}")
    return mgr


def drain_free(pool, free_q, stop_evt):
    while not stop_evt.is_set():
        drained = 0
        try:
            while True:
                sid = free_q.get_nowait()
                pool.release(sid)
                drained += 1
        except queue.Empty:
            pass
        if drained == 0:
            time.sleep(0.001)


# ---------- Main run loop ----------

def run_server(cfg: Config):
    import multiprocessing as mp
    ipc_host, ipc_port, ipc_key = cfg.data_server.ipc_host, cfg.data_server.ipc_port, cfg.data_server.ipc_key
    pool_size = cfg.data_server.pool_size
    num_workers = cfg.data_server.num_workers
    batch_size = cfg.diloco.batch_size_per_step
    work_q = mp.Queue(maxsize=pool_size)  # to trainers: descriptors
    free_q = mp.Queue(maxsize=pool_size)  # from trainers: slot ids
    mgr = start_manager(work_q, free_q, address=(ipc_host, ipc_port), authkey=ipc_key.encode())

    pool = SharedBatchPool(pool_size=pool_size)

    cfn = collate_fn(cfg.data)

    stop_evt = threading.Event()
    threading.Thread(target=drain_free, args=(pool, free_q, stop_evt), daemon=True).start()

    def _handle_signal(signum, frame):
        logger.info(f"Data server received signal {signum}, shutting down gracefully...")
        stop_evt.set()
        raise KeyboardInterrupt("Signal received")

    # Ensure graceful shutdown on SIGINT/SIGTERM to free the port
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_evt.is_set():
            train_ds, _ = get_train_val_datasets(cfg.data)

            ds = BufferedShuffleIterable(train_ds, buffer_size=cfg.data.shuffle_buffer_size, seed=0)
            loader = DataLoader(
                ds,                                 # yields (uid, sample) or sample
                batch_size=batch_size or cfg.diloco.batch_size_per_step,
                num_workers=num_workers,
                pin_memory=False,                         # copy to SHM, pinning unnecessary here
                drop_last=True,
                shuffle=False,                            # shuffle ignored for IterableDataset
                collate_fn=cfn,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )
            for i, batch in enumerate(loader):
                if stop_evt.is_set():
                    break
                logger.info(f"[server] batch {i}: work_q:{work_q.qsize()} free_slots:{pool.free.qsize()}")

                uids, batch = batch
                desc = pool.put_batch(batch["inputs"], batch["labels"])
                
                while True:
                    if stop_evt.is_set():
                        break
                    try:
                        work_q.put(desc, timeout=0.01)
                        break
                    except queue.Full:
                        pass
    finally:
        pool.shutdown()
        try:
            work_q.cancel_join_thread()
            free_q.cancel_join_thread()
        except Exception:
            pass

        mgr.shutdown()

# ---------- CLI ----------

if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*is used more than once. Remove its duplicate as parameters should be unique.*")
        res = parse_args(standalone_mode=False)
        if isinstance(res, int):
            quit()
        elif isinstance(res, tuple):
            cfg, extra_kwargs = res
            run_server(cfg)
        else:
            raise ValueError(f"Unexpected return type: {type(res)}")
