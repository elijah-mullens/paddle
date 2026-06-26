import os
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d

import snapy


def _register_ucx_backend() -> None:
    try:
        import commux
    except ImportError as exc:
        raise RuntimeError(
            "UCX backend requires commux. Install commux or use backend='gloo'."
        ) from exc
    commux.register()


def start_dist(backend: str) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", os.environ["RANK"])

    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = int(os.environ.get("DEVICE_ID", local_rank))
    if backend == "gloo":
        dist.init_process_group(backend="gloo", init_method="env://")
        device = torch.device("cpu")
    elif backend == "ucx":
        _register_ucx_backend()
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend="ucx", init_method="env://")
    else:
        raise ValueError("Unsupported backend")

    snapy.distributed.set_process_group(dist_c10d._get_default_group())
    return device


def close_dist():
    if dist.is_initialized():
        dist.destroy_process_group()
