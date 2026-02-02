import os
import torch
import torch.distributed as dist

def main():
    # torchrun sets these env vars
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(
        f"hello from rank={rank}/{world_size} local_rank={local_rank} "
        f"hostname={os.uname().nodename}", flush=True)

    dist.init_process_group(backend="gloo", init_method="env://")
    print("init done", flush=True)

    x = torch.tensor([1+rank], dtype=torch.float32)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    print(f"all_reduce_sum={x.item()}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    print("AAAAAAAAAAAAAA", flush=True)
    main()
