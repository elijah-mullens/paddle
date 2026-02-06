#!/bin/bash
#SBATCH --job-name=torchrun-3x2
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1              # one torchrun per node
#SBATCH --gres=gpu:2                     # 2 GPUs per node
#SBATCH --cpus-per-task=16               # adjust
#SBATCH --mem=0                          # or set a value like 64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# ---- User settings ----
SIF=/path/to/your_image.sif
WORKDIR=/path/to/your/project               # host path
TRAIN=/workspace/train.py                   # path *inside* container
PORT=29500

# (Optional) bind any datasets/checkpoints
BIND_OPTS="--bind ${WORKDIR}:/workspace"

# ---- Derive rendezvous info from Slurm allocation ----
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${PORT}"
export NNODES="${SLURM_JOB_NUM_NODES}"
export GPUS_PER_NODE=2
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}  GPUS_PER_NODE=${GPUS_PER_NODE}  WORLD_SIZE=${WORLD_SIZE}"
echo "NODELIST=${SLURM_JOB_NODELIST}"

# ---- Helpful NCCL/transport knobs (adjust to your cluster) ----
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_DISABLE=1                 # uncomment if no InfiniBand
# export NCCL_SOCKET_IFNAME=ib0            # or eno1/bond0/etc
# export GLOO_SOCKET_IFNAME=ib0

# ---- Launch one torchrun per node; torchrun spawns 2 procs on that node ----
srun --nodes=${NNODES} --ntasks=${NNODES} --ntasks-per-node=1 \
  singularity exec --nv ${BIND_OPTS} "${SIF}" bash -lc '
    set -euo pipefail

    echo "Running on $(hostname)  SLURM_PROCID=${SLURM_PROCID}"
    export NODE_RANK=${SLURM_PROCID}

    torchrun \
      --nnodes='"${NNODES}"' \
      --nproc_per_node='"${GPUS_PER_NODE}"' \
      --node_rank=${NODE_RANK} \
      --rdzv_backend=c10d \
      --rdzv_endpoint='"${MASTER_ADDR}:${MASTER_PORT}"' \
      '"${TRAIN}"' \
      --data_dir /workspace/data \
      --output_dir /workspace/out
  '
