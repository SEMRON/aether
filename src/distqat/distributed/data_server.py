import warnings
import os
import click
import queue, threading, time
import random

import gymnasium as gym

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
from distqat.distributed.model import SwarmBaselineModel
from distqat.utils.buffer import RolloutBuffer


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
        slot = self.slots[slot_id] or {}
        if slot.get(key) is None:
            slot[key] = None
        shm = slot[key]
        if shm is None or shm.size < nbytes:
            if shm is not None:
                try: shm.close(); shm.unlink()
                except: pass
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            slot[key] = shm
            self.slots[slot_id] = slot
        return shm

    def put_batch(self, inputs, labels: torch.Tensor):
        """
        Store inputs and labels in shared memory.
        
        Args:
            inputs: Either a torch.Tensor (for regular data) or a dict with keys like 
                   "x" and "edge_index" (for graph data)
            labels: torch.Tensor of labels
        """
        slot_id = self.free.get()  # blocks if none free
        
        if isinstance(inputs, tuple):
            # Tuple data: store each tensor under inputs_0, inputs_1, ...
            inputs_meta = []
            for i, tensor in enumerate(inputs):
                t = tensor.detach().contiguous().cpu()
                nbytes = t.element_size() * t.nelement()
                shm = self._ensure(slot_id, f"inputs_{i}", nbytes)
                t_bytes = t.numpy().view(np.uint8).reshape(-1)
                np.frombuffer(shm.buf, dtype=np.uint8, count=nbytes)[:] = t_bytes
                inputs_meta.append({"name": shm.name, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")})

        else:
            # Regular data: inputs is a single tensor
            x = inputs.detach().contiguous().cpu()
            x_nbytes = x.element_size() * x.nelement()
            x_shm = self._ensure(slot_id, "inputs", x_nbytes)
            x_bytes = x.numpy().view(np.uint8).reshape(-1)
            np.frombuffer(x_shm.buf, dtype=np.uint8, count=x_nbytes)[:] = x_bytes
            inputs_meta = {"name": x_shm.name, "shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", "")}

        # Handle labels
        y = labels.detach().contiguous().cpu()
        y_nbytes = y.element_size() * y.nelement()
        y_shm = self._ensure(slot_id, "labels", y_nbytes)
        y_bytes = y.numpy().view(np.uint8).reshape(-1)
        np.frombuffer(y_shm.buf, dtype=np.uint8, count=y_nbytes)[:] = y_bytes

        return {
            "slot": slot_id,
            "inputs": inputs_meta,
            "labels": {"name": y_shm.name, "shape": list(y.shape), "dtype": str(y.dtype).replace("torch.", "")},
        }

    def release(self, slot_id: int):
        self.free.put(slot_id)

    def shutdown(self):
        for s in self.slots:
            if not s: continue
            for k in s.keys():
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



# ---------- RL specific functions ----------
def rl_iteration(cfg, global_step):
    pass


def monitor_iter(stop_iter_evt, work_q, pool):
    while not stop_iter_evt.is_set():
        time.sleep(0.5)
        logger.info(f"[server] work_q:{work_q.qsize()} free_slots:{pool.free.qsize()}")
        # dht.get(f"{cfg.experiment_prefix}_", latest=True)



def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # Gymnasium requires an explicit observation_space for TransformObservation in newer versions.
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


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

    cfn = collate_fn(cfg.data, cfg.model_pipeline.pipeline[0])

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
        if cfg.data.task_type == "rl":
            seed = 0
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            # env setup
            envs = gym.vector.SyncVectorEnv(
                [
                    make_env(
                        cfg.data.dataset_name,
                        i,
                        False,  # capture_video removed (not in distqat config)
                        cfg.experiment_prefix,
                        float(cfg.ppo.gamma),
                    )
                    for i in range(int(cfg.ppo.num_envs))
                ]
            )
            assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
            # while not stop_evt.is_set():
            
            global_step = 0
            # ALGO Logic: Storage setup
            num_steps = int(cfg.ppo.num_steps) # default:2048
            update_epochs = int(cfg.ppo.update_epochs) # default: 64
            # num_envs = int(cli.num_servers) # default:1
            num_envs = int(cfg.ppo.num_envs)
            batch_size = int(num_envs * num_steps) # default:2048
            minibatch_size = int(cfg.ppo.minibatch_size) # num_minibatches = batch_size_per_step, default: 32; num_minibatches = batch_size / batch_size_per_step = 64
            num_iterations = int(cfg.ppo.total_timesteps / batch_size)
            device = cfg.device
            print("initial peers: ", cfg.network.initial_peers)
            
            
            rollout = RolloutBuffer(
                num_steps=num_steps,
                num_envs=num_envs,
                obs_shape=tuple(envs.single_observation_space.shape),
                action_shape=tuple(envs.single_action_space.shape),
                device=device,
                in_dim=cfg.model_pipeline.pipeline[0].in_dim,
                action_dim=cfg.model_pipeline.pipeline[0].action_dim,
            )

            next_obs, _ = envs.reset(seed=seed)
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            next_done = torch.zeros(num_envs).to(device)

            # trainer = SwarmTrainer(config=cfg, trainer_id=-2, use_baseline_model=True)
            # model = trainer.model
            model = SwarmBaselineModel(config=cfg, trainer_id=-2, disable_quant=True)
            # model = SwarmModel(config=cfg, trainer_id=-2)
            model.to(device)
            # dht = model.dht
            agent = model.model.model_pipeline[0]
            log_episodic_from_infos = model.metrics_logger.log_episodic_from_infos
            step = 0
            for iteration in range(1, num_iterations + 1):
                stop_iter_evt = threading.Event()
                threading.Thread(target=monitor_iter, args=(stop_iter_evt, work_q, pool), daemon=True).start()

                num_episodes = 0

                # Update policy for the next iteration
                model.optimizer.load_state_from_peers()
                rollout.reset()
                for _ in range(0, num_steps):
                    global_step, next_obs, next_done = rollout.collect_and_add_step(
                        agent=agent,
                        envs=envs,
                        global_step=global_step,
                        next_obs=next_obs,
                        next_done=next_done,
                        log_episodic_from_infos=log_episodic_from_infos,
                    )

                advantages, returns = agent.compute_gae_returns(
                    next_obs=next_obs,
                    next_done=next_done,
                    rewards=rollout.rewards,
                    dones=rollout.dones,    
                    values=rollout.values,
                    gamma=float(cfg.ppo.gamma),
                    gae_lambda=float(cfg.ppo.gae_lambda),
                )
                rollout.set_advantages_and_returns(advantages, returns)
                b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = rollout.get_batch()
                # Enqueue ONE PPO minibatch per queue item, for a total of
                # update_epochs * (batch_size / minibatch_size) updates.
                if batch_size % minibatch_size != 0:
                    raise ValueError(
                        f"batch_size ({batch_size}) must be divisible by minibatch_size ({minibatch_size})"
                    )
                num_minibatches = batch_size // minibatch_size

                for epoch in range(update_epochs):
                    if stop_iter_evt.is_set() or stop_evt.is_set():
                        break

                    # Put indices on the same device as b_* tensors so indexing always works.
                    b_inds = torch.randperm(batch_size, device=b_obs.device)

                    for start in range(0, batch_size, minibatch_size):
                        if stop_iter_evt.is_set() or stop_evt.is_set():
                            break

                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]

                        mb_inputs = (
                            b_obs[mb_inds],
                            b_logprobs[mb_inds],
                            b_actions[mb_inds],
                            b_advantages[mb_inds],
                            b_returns[mb_inds],
                            b_values[mb_inds],
                        )

                        # Carry env-scale step for logging on the trainer side
                        labels = torch.zeros((1,), dtype=torch.float32, device=b_obs.device)
                        labels[0] = float(global_step)

                        desc = pool.put_batch(mb_inputs, labels)

                        while True:
                            if stop_iter_evt.is_set() or stop_evt.is_set():
                                break
                            try:
                                work_q.put(desc, timeout=0.01)
                                break
                            except queue.Full:
                                pass

                        step += 1
        else:
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
