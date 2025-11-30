import hashlib, heapq
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Sequence, Set, Union, Tuple
from hivemind import DHT
from hivemind.utils import get_dht_time
import torch


ACTIVE_KEY = "{run_id}:active_processes"

def register_process(dht: DHT, run_id: str, ttl: float = 30.0) -> None:
    # Add to the active set (best-effort, idempotent)
    print(f"Registering process with run_id: {run_id}")
    key = ACTIVE_KEY.format(run_id=run_id)
    peer_id = dht.peer_id.to_string()
    data, _ = dht.get(key, latest=True) or ([], None)
    active: List[str] = list(data) if isinstance(data, list) else []
    if peer_id not in active:
        active.append(peer_id)
    # store with TTL so dead peers eventually disappear if you refresh periodically
    dht.store(key=key, value=sorted(active), expiration_time=get_dht_time() + ttl, return_future=True)

def load_active_set(dht: DHT, run_id: str) -> List[str]:
    key = ACTIVE_KEY.format(run_id=run_id)
    peer_id = dht.peer_id.to_string()
    data, _ = dht.get(key, latest=True) or (None, None)
    if not data:
        # bootstrap: I'm the first
        dht.store(key=key, value=[peer_id], expiration_time=get_dht_time() + 300.0, return_future=True)
        return [peer_id]
    return list(data)

def get_outer_step(dht, prefix) -> int:
    v, _ = dht.get(f"{prefix}:outer_step", latest=True) or (None, None)
    return int(v) if isinstance(v, int) else 0

def set_outer_step_if_greater(dht, prefix, new_step: int, ttl: float = 600.0) -> None:
    # simple set-if-greater using a read-then-write; races are benign (eventual max wins)
    curr = get_outer_step(dht, prefix)
    if new_step > curr:
        dht.store(
            key=f"{prefix}:outer_step",
            value=int(new_step),
            expiration_time=get_dht_time() + ttl,
            return_future=False,  # wait for completion
        )

class ElasticOwnershipFilter(IterableDataset):
    def __init__(self, ds, epoch_seed_int, outer_step, rank, k, max_emit):
        self.ds = ds
        self.rank = rank
        self.k = k
        self.max_emit = max_emit
        N = len(self.ds)

        # permutation stable for the whole epoch
        g = torch.Generator().manual_seed(epoch_seed_int)
        self.perm = torch.randperm(N, generator=g)

        emit_per_peer = max_emit
        self.global_base = (outer_step * k * emit_per_peer) % N

    def __iter__(self):
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        wnum = 1 if wi is None else wi.num_workers

        s = wid
        emitted = 0
        N = len(self.ds)
        while emitted < self.max_emit:
            t = self.global_base + self.rank + self.k * s
            i = int(self.perm[t % N])
            yield i, self.ds[i]
            s += wnum
            emitted += 1
