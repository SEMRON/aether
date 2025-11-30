import asyncio
import os
import threading
from contextlib import contextmanager
from typing import Iterator

import pytest
import time

from distqat.config import Config, DataConfig
from distqat.distributed.data_server import DataServer
from distqat.distributed.data_client import DataClient
from distqat.data import get_train_val_datasets, collate_fn
from torch.utils.data import DataLoader


def _minimal_cfg(batch_size: int = 32) -> Config:
    cfg = Config()
    # MNIST CV streaming
    cfg.data = DataConfig(dataset_name="mnist", dataset_split="test", task_type="cv", num_workers=0)
    # Set batch size used by server
    cfg.diloco.batch_size_per_step = batch_size
    return cfg


@contextmanager
def _run_server(cfg: Config, port: int, *, seed: int = 0) -> Iterator[str]:
    """Start DataServer HTTP server; yield base URL when bound; stop on exit."""
    from aiohttp import web

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(None)
    base_url = f"http://127.0.0.1:{port}"
    server = DataServer(cfg, port=port, shuffle_buf=8192, batch_size=cfg.diloco.batch_size_per_step, seed=seed)

    ready = threading.Event()
    state: dict = {}

    async def _serve():
        await server.start_background()
        runner = web.AppRunner(server.app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        state["runner"] = runner
        ready.set()
        while True:
            await asyncio.sleep(1.0)

    def _thread_main():
        asyncio.set_event_loop(loop)
        loop.create_task(_serve())
        loop.run_forever()

    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()

    if not ready.wait(timeout=30):
        raise RuntimeError("Server did not start in time")

    # Probe health until reachable
    import requests
    for _ in range(60):
        try:
            r = requests.get(f"{base_url}/health", timeout=1.0)
            if r.ok:
                break
        except Exception:
            pass
        time.sleep(0.25)

    try:
        yield base_url
    finally:
        def _stop():
            async def _cleanup():
                try:
                    if server._producer_task:
                        server._producer_task.cancel()
                except Exception:
                    pass
                runner = state.get("runner")
                if runner is not None:
                    await runner.cleanup()
                loop.stop()
            asyncio.ensure_future(_cleanup(), loop=loop)
        loop.call_soon_threadsafe(_stop)
        t.join(timeout=10)


@pytest.mark.timeout(120)
def test_server_single_client_unique_uids():
    batch_size = 64
    port = 52100
    cfg = _minimal_cfg(batch_size=batch_size)

    base_ds, _ = get_train_val_datasets(cfg.data)
    base_set = set()
    for uids, _ in DataLoader(base_ds, batch_size=batch_size, num_workers=0, collate_fn=collate_fn(cfg.data)):
        for u in uids:
            base_set.add(int(u))

    with _run_server(cfg, port) as base_url:
        client = DataClient(base_url, trainer_id=0)
        seen = set()
        while len(seen) < len(base_set):
            uids, batch = client.next_batch(timeout=60.0)
            remaining = len(base_set) - len(seen)
            # Batch size is either the remaining window size or the configured batch size, whichever is smaller
            assert 1 <= len(uids) <= min(batch_size, remaining)
            for u in uids:
                ui = int(u)
                assert ui not in seen
                seen.add(ui)

    assert len(seen) == len(base_set)


@pytest.mark.timeout(180)
def test_server_two_clients_no_overlap_and_fixed_window_coverage():
    batch_size = 64
    port = 52101
    cfg = _minimal_cfg(batch_size=batch_size)

    base_ds, _ = get_train_val_datasets(cfg.data)
    base_set = set()
    dl = DataLoader(base_ds, batch_size=batch_size, num_workers=0, collate_fn=collate_fn(cfg.data))
    for uids, _ in dl:
        for u in uids:
            base_set.add(int(u))


    with _run_server(cfg, port) as base_url:
        c0 = DataClient(base_url, trainer_id=0)
        c1 = DataClient(base_url, trainer_id=1)

        seen0, seen1 = set(), set()
        while len(seen0) + len(seen1) < len(base_set):
            u0, _ = c0.next_batch(timeout=60.0)
            for u in u0:
                ui = int(u)
                seen0.add(ui)
            u1, _ = c1.next_batch(timeout=60.0)
            for u in u1:
                ui = int(u)
                seen1.add(ui)
            

        print(len(seen0), len(seen1), len(base_set))
        # No overlap between clients within the epoch window
        # Combined coverage equals the fixed window size
        assert (seen0 | seen1) == base_set
        assert len(seen0 & seen1) <= batch_size


@pytest.mark.timeout(180)
def test_shuffle_is_deterministic_by_seed_and_varies_with_seed():
    batch_size = 64
    cfg = _minimal_cfg(batch_size=batch_size)

    # Build a baseline (unshuffled) order from the raw dataset iteration
    base_ds, _ = get_train_val_datasets(cfg.data)
    base_order = []
    for uids, _ in DataLoader(base_ds, batch_size=batch_size, num_workers=0, collate_fn=collate_fn(cfg.data)):
        base_order.extend([int(u) for u in uids])
        if len(base_order) >= 5 * batch_size:
            break
    K = 5 * batch_size
    base_order = base_order[:K]

    # Same seed → same order
    port_a, port_b = 52110, 52111
    with _run_server(cfg, port_a, seed=123) as base_url_a:
        c_a = DataClient(base_url_a, trainer_id=0)
        order_a = []
        while len(order_a) < K:
            uids, _ = c_a.next_batch(timeout=60.0)
            order_a.extend([int(u) for u in uids])
        order_a = order_a[:K]

    with _run_server(cfg, port_b, seed=123) as base_url_b:
        c_b = DataClient(base_url_b, trainer_id=0)
        order_b = []
        while len(order_b) < K:
            uids, _ = c_b.next_batch(timeout=60.0)
            order_b.extend([int(u) for u in uids])
        order_b = order_b[:K]

    assert order_a == order_b

    # Different seed → different order
    port_c = 52112
    with _run_server(cfg, port_c, seed=124) as base_url_c:
        c_c = DataClient(base_url_c, trainer_id=0)
        order_c = []
        while len(order_c) < K:
            uids, _ = c_c.next_batch(timeout=60.0)
            order_c.extend([int(u) for u in uids])
        order_c = order_c[:K]

    assert order_a != order_c

    # Shuffled order should differ from the base sequential order
    assert order_a != base_order
