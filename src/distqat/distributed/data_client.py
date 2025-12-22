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
        inputs_meta, ymeta = desc["inputs"], desc["labels"]
        
        if isinstance(inputs_meta, list):
            # Tuple data: inputs_meta is a list of tensor descriptors (inputs_0, inputs_1, ...)
            tensors = []
            shm_handles = []
            for meta in inputs_meta:
                t = _from_shm(meta["name"], meta["shape"], meta["dtype"])
                tensors.append(t)
                shm_handles.append(t._shm)
            inputs = tuple(tensors)
        else:
            # Regular data: inputs_meta is a single tensor descriptor
            x = _from_shm(inputs_meta["name"], inputs_meta["shape"], inputs_meta["dtype"])
            inputs = x
            shm_handles = [x._shm]
        
        y = _from_shm(ymeta["name"], ymeta["shape"], ymeta["dtype"])
        shm_handles.append(y._shm)
        
        def release():
            # return slot id to server and close local handles
            self.free_q.put(desc["slot"])
            # prevent resource_tracker from trying to unlink shared segments we only attached to
            try:
                for shm in shm_handles:
                    resource_tracker.unregister(shm._name, 'shared_memory')
                    shm.close()
            except Exception:
                pass
        return {"inputs": inputs, "labels": y}, release
