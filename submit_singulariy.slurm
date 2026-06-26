#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---- User settings ----
SIF=<SIF_FILE>
PORT=29500

# ---- Run settings ----
EXE=<PYTHON_RUN_FILE>
CONFIG=<CONFIG_YAML_FILE>
OUTPUT_DIR=<OUTPUT_DIR>

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

# ---- Helpful UCX/transport knobs (adjust to your cluster) ----
# export UCX_LOG_LEVEL=INFO
# export UCX_WARN_UNUSED_ENV_VARS=n
# export UCX_TLS=tcp                # uncomment if no InfiniBand
# export UCX_NET_DEVICES=ib0            # or eno1/bond0/etc
# export GLOO_SOCKET_IFNAME=ib0

# ---- Launch one torchrun per node; torchrun spawns 2 procs on that node ----
srun --nodes=${NNODES} --ntasks=${NNODES} --ntasks-per-node=1 \
  singularity exec --nv --pwd "$PWD" "${SIF}" bash -lc '
    set -euo pipefail

    echo "Running on $(hostname)  SLURM_PROCID=${SLURM_PROCID}"
    export NODE_RANK=${SLURM_PROCID}

    source /opt/venv/bin/activate

    torchrun \
      --nnodes='"${NNODES}"' \
      --nproc_per_node='"${GPUS_PER_NODE}"' \
      --node_rank=${NODE_RANK} \
      --master_addr='"${MASTER_ADDR}"' \
      --master_port='"${MASTER_PORT}"' \
      '"${EXE}"' --config='"${CONFIG}"' --output-dir='"${OUTPUT_DIR}"'
  '
