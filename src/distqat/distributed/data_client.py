from multiprocessing.managers import BaseManager
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
from queue import Full
import numpy as np
import torch

class ServerManager(BaseManager): pass


def _from_shm(name: str, shape, dtype_str: str):
    shm = shared_memory.SharedMemory(name=name)
    np_view = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    t = torch.from_numpy(np_view)  # shares memory, no copy
    t._shm = shm                    # keep handle alive
    return t

class DataClient:
    def __init__(self, host="127.0.0.1", port=52555, authkey=b"distqat"):
        ServerManager.register("get_work_q")
        ServerManager.register("get_free_q")
        self.mgr = ServerManager(address=(host, port), authkey=authkey)
        self.mgr.connect()
        self.work_q = self.mgr.get_work_q()
        self.free_q = self.mgr.get_free_q()

    def next_batch(self):
        desc = self.work_q.get()  # blocks
        xmeta, ymeta = desc["inputs"], desc["labels"]
        x = _from_shm(xmeta["name"], xmeta["shape"], xmeta["dtype"])
        y = _from_shm(ymeta["name"], ymeta["shape"], ymeta["dtype"])
        def release():
            # return slot id to server and close local handles
            self.free_q.put(desc["slot"])
            # prevent resource_tracker from trying to unlink shared segments we only attached to
            try:
                resource_tracker.unregister(x._shm._name, 'shared_memory')
                resource_tracker.unregister(y._shm._name, 'shared_memory')
                x._shm.close()
                y._shm.close()
            except Exception:
                pass
        return {"inputs": x, "labels": y}, release
