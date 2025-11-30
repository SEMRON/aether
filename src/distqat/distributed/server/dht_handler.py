import threading
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

from hivemind.dht import DHT, DHTNode, DHTValue
from hivemind.utils import DHTExpiration, get_dht_time


from distqat.distributed.client.expert import RemoteExpert
from distqat.distributed.server.expert_uid import (
    FLAT_EXPERT,
    UID_DELIMITER,
    UID_PATTERN,
    Coordinate,
    ExpertPrefix,
    ExpertUID,
    is_valid_uid,
    split_uid,
)
from distqat.distributed.utils.networking import Endpoint, get_port


class DHTHandlerThread(threading.Thread):
    """
    A background thread that periodically publishes expert information to the DHT.
    
    This thread runs continuously and updates the DHT with information about local experts,
    making them discoverable by other nodes in the SWARM network.
    
    :param experts: dictionary mapping expert UIDs to expert instances
    :param dht: DHT instance to publish expert information to
    :param endpoint: network endpoint where this server can be reached
    :param update_period: how often (in seconds) to update expert information in DHT
    :param kwargs: additional arguments passed to threading.Thread
    """
    def __init__(self, experts, dht: DHT, endpoint: Endpoint, update_period: int = 5, **kwargs):
        super().__init__(**kwargs)
        assert get_port(endpoint) is not None
        self.endpoint = endpoint
        self.experts = experts
        self.dht = dht
        self.update_period = update_period
        self.stop = threading.Event()

    def run(self) -> None:
        """
        Main thread loop that periodically declares experts in the DHT.
        
        Continuously publishes expert information to the DHT at regular intervals
        until the thread is signaled to stop.
        """
        declare_experts(self.dht, self.experts.keys(), self.endpoint)
        while not self.stop.wait(self.update_period):
            declare_experts(self.dht, self.experts.keys(), self.endpoint)


def declare_experts(
    dht: DHT, uids: Sequence[ExpertUID], endpoint: Endpoint, expiration: DHTExpiration = 60, wait: bool = True
) -> Dict[ExpertUID, bool]:
    """
    Make experts visible to all DHT peers; update timestamps if declared previously.

    :param dht: DHT instance to store expert information in
    :param uids: a list of expert ids to update
    :param endpoint: endpoint that serves these experts, usually your server endpoint (e.g. "201.111.222.333:1337")
    :param wait: if True, awaits for declaration to finish, otherwise runs in background
    :param expiration: experts will be visible for this many seconds
    :returns: if wait, returns store status for every key (True = store succeeded, False = store rejected)
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    for uid in uids:
        assert is_valid_uid(uid), f"{uid} is not a valid expert uid. All uids must follow {UID_PATTERN.pattern}"
    return dht.run_coroutine(
        partial(_declare_experts, uids=list(uids), endpoint=endpoint, expiration=expiration), return_future=not wait
    )


async def _declare_experts(
    dht: DHT, node: DHTNode, uids: List[ExpertUID], endpoint: Endpoint, expiration: DHTExpiration
) -> Dict[ExpertUID, bool]:
    """
    Internal async function to store expert UIDs and endpoints in the DHT.
    
    Creates DHT entries for each expert UID and its hierarchical prefixes to enable
    efficient expert discovery across the SWARM network.
    
    :param dht: DHT instance (used for configuration)
    :param node: DHT node to perform store operations
    :param uids: list of expert UIDs to declare
    :param endpoint: network endpoint where experts can be reached
    :param expiration: expiration time for DHT entries
    :returns: dictionary mapping UIDs to store success status
    """
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    expiration_time = get_dht_time() + expiration
    data_to_store: Dict[Tuple[ExpertPrefix, Optional[Coordinate]], DHTValue] = {}
    for uid in uids:
        data_to_store[uid, None] = endpoint
        prefix = uid if uid.count(UID_DELIMITER) > 1 else f"{uid}{UID_DELIMITER}{FLAT_EXPERT}"
        for i in range(prefix.count(UID_DELIMITER) - 1):
            prefix, last_coord = split_uid(prefix)
            data_to_store[prefix, last_coord] = [uid, endpoint]

    keys, maybe_subkeys, values = zip(*((key, subkey, value) for (key, subkey), value in data_to_store.items()))
    store_ok = await node.store_many(keys, values, expiration_time, subkeys=maybe_subkeys, num_workers=num_workers)
    return store_ok


def remove_experts(
    dht: DHT, uids: List[ExpertUID], *, wait: bool = True
) -> Dict[ExpertUID, bool]:
    """
    Expire expert UIDs immediately across the DHT using a single multi-key operation.

    :param dht: DHT instance
    :param uids: list of expert uids to remove
    :param wait: if True, waits for completion and returns store status; otherwise runs in background
    :returns: mapping of uid -> store status if wait else MPFuture
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    return dht.run_coroutine(partial(_remove_experts, uids=list(uids)), return_future=not wait)


async def _remove_experts(
    dht: DHT, node: DHTNode, uids: List[ExpertUID]
) -> Dict[ExpertUID, bool]:
    """
    Internal async function to set past expiration for UIDs using store_many.
    """
    expiration_time = get_dht_time() - 1
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    # Expire main UID keys only; discovery uses exact UID lookups
    keys = uids
    values: List[Optional[DHTValue]] = [None] * len(uids)
    store_ok = await node.store_many(keys, values, expiration_time, subkeys=None, num_workers=num_workers)
    return store_ok



def store_expert_batch_size(
    dht: DHT, expert_uid: ExpertUID, batch_size: int, expiration: DHTExpiration = 60, wait: bool = True
) -> bool:
    """
    Store batch_size_per_step for an expert in the DHT.
    
    :param dht: DHT instance to store batch size in
    :param expert_uid: expert UID to store batch size for
    :param batch_size: batch_size_per_step value
    :param expiration: expiration time in seconds
    :param wait: if True, awaits for storage to finish
    :returns: True if storage succeeded, False otherwise
    """
    key = f"{expert_uid}_batch_size"
    expiration_time = get_dht_time() + expiration
    return dht.store(key, str(batch_size), expiration_time=expiration_time, return_future=not wait)


def get_expert_batch_size(
    dht: DHT, expert_uid: ExpertUID, latest: bool = True
) -> Optional[int]:
    """
    Retrieve batch_size_per_step for an expert from the DHT.
    
    :param dht: DHT instance to query
    :param expert_uid: expert UID to get batch size for
    :param latest: if True, get the latest value
    :returns: batch_size_per_step if found, None otherwise
    """
    key = f"{expert_uid}_batch_size"
    result = dht.get(key, latest=latest)
    if result is not None and result.value is not None:
        try:
            return int(result.value)
        except (ValueError, TypeError):
            return None
    return None


def get_experts(
    dht: DHT, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration] = None, return_future: bool = False
) -> List[Optional[RemoteExpert]]:
    """
    :param dht: DHT instance to search for experts in
    :param uids: find experts with these ids from across the DHT
    :param expiration_time: if specified, return experts that expire no sooner than this (based on get_dht_time)
    :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
    :returns: a list of [RemoteExpert if found else None]
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    return dht.run_coroutine(partial(_get_experts, uids=list(uids), expiration_time=expiration_time), return_future)


async def _get_experts(
    dht: DHT, node: DHTNode, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration]
) -> List[Optional[RemoteExpert]]:
    """
    Internal async function to retrieve expert information from the DHT.
    
    Looks up expert UIDs in the DHT and creates RemoteExpert instances for
    experts that are found and not expired.
    
    :param dht: DHT instance (used for configuration)
    :param node: DHT node to perform lookup operations
    :param uids: list of expert UIDs to look up
    :param expiration_time: minimum expiration time for valid entries
    :returns: list of RemoteExpert instances (None for missing experts)
    """
    if expiration_time is None:
        expiration_time = get_dht_time()
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    found: Dict[ExpertUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

    experts: List[Optional[RemoteExpert]] = [None] * len(uids)
    for i, uid in enumerate(uids):
        if found[uid] is not None and isinstance(found[uid].value, Endpoint):
            experts[i] = RemoteExpert(uid, found[uid].value)
    return experts
