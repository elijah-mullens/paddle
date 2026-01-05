#!/bin/bash

# Pick master addr/port
MASTER_ADDR=$(hostname)
MASTER_PORT=29500

srun --mpi=none -n 2 bash -c '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export NODE_RANK=$SLURM_NODEID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'

  echo "Launching rank $RANK/$WORLD_SIZE on node $(hostname), local_rank=$LOCAL_RANK"

  python ./straka.py
'
