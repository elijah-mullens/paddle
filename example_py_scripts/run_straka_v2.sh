#!/usr/bin/env bash
set -euo pipefail

# Pick master addr/port (force IPv4)
MASTER_ADDR="$(hostname -I | awk '{print $1}')"
MASTER_PORT=29500

# Optional: force gloo interface (good on clusters)
# IFACE="$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if ($i=="dev") print $(i+1)}')"
# export GLOO_SOCKET_IFNAME="$IFACE"

srun --mpi=none -n 2 bash -c "
  export RANK=\$SLURM_PROCID
  export WORLD_SIZE=\$SLURM_NTASKS
  export LOCAL_RANK=\$SLURM_LOCALID
  export NODE_RANK=\$SLURM_NODEID

  export MASTER_ADDR=$MASTER_ADDR
  export MASTER_PORT=$MASTER_PORT

  echo \"Launching rank \$RANK/\$WORLD_SIZE on node \$(hostname), local_rank=\$LOCAL_RANK, master=$MASTER_ADDR:$MASTER_PORT\"

  python ./straka.py
"
