#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 TRAIN_SCRIPT [TRAIN_ARGS...]"
  echo
  echo "Example:"
  echo "  $0 run_jupiter_moist.py -c jupiter_gcm_H2O-NH3_F100.yaml --output-dir output_6gpu"
  echo
  echo "Optional environment variables:"
  echo "  PADDLE_NODES='dart2 dart3 dart1'"
  echo "  GPUS_PER_NODE=2"
  echo "  MASTER_ADDR=dart2"
  echo "  MASTER_PORT=29500"
  echo "  LOCAL_NODE=\${PADDLE_NODES%% *}  # node alias for the submit host"
  echo "  DIST_SOCKET_IFNAME=<routable interface>  # exports GLOO/NCCL/UCX interface vars"
  echo "  WORKDIR=$(pwd)  # defaults to the directory where this launcher is submitted"
  exit 2
fi

read -r -a NODES <<< "${PADDLE_NODES:-dart2 dart3 dart1}"
NNODES=${#NODES[@]}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
MASTER_ADDR=${MASTER_ADDR:-${NODES[0]}}
MASTER_PORT=${MASTER_PORT:-29500}
SUBMIT_DIR=$(pwd)
WORKDIR=${WORKDIR:-${SUBMIT_DIR}}
LOCAL_HOST=$(hostname)
LOCAL_SHORT_HOST=$(hostname -s)
LOCAL_NODE=${LOCAL_NODE:-${NODES[0]}}
DIST_SOCKET_IFNAME=${DIST_SOCKET_IFNAME:-}

if [[ ${NNODES} -ne 3 ]]; then
  echo "Expected exactly 3 nodes, got ${NNODES}: ${NODES[*]}" >&2
  exit 2
fi

if [[ ${GPUS_PER_NODE} -ne 2 ]]; then
  echo "Expected 2 GPUs per node, got GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
  exit 2
fi

TRAIN_SCRIPT=$1
shift

quote_args() {
  local quoted=()
  for arg in "$@"; do
    quoted+=("$(printf "%q" "${arg}")")
  done
  printf "%s " "${quoted[@]}"
}

is_local_node() {
  local node=$1

  if [[ "${node}" == "${LOCAL_NODE}" || "${node}" == "${LOCAL_HOST}" || "${node}" == "${LOCAL_SHORT_HOST}" ]]; then
    return 0
  fi

  if command -v getent >/dev/null 2>&1; then
    local local_ips node_ips
    local_ips=$(hostname -I 2>/dev/null || true)
    node_ips=$(getent ahosts "${node}" 2>/dev/null | awk '{print $1}' | sort -u || true)
    for node_ip in ${node_ips}; do
      for local_ip in ${local_ips}; do
        if [[ "${node_ip}" == "${local_ip}" ]]; then
          return 0
        fi
      done
    done
  fi

  return 1
}

TRAIN_CMD="$(quote_args "${TRAIN_SCRIPT}" "$@")"
REMOTE_PREFIX='if [[ -f "${HOME}/.bash_profile" ]]; then source "${HOME}/.bash_profile"; fi'

echo "Launching ${NNODES} nodes x ${GPUS_PER_NODE} GPUs = $((NNODES * GPUS_PER_NODE)) processes"
echo "Nodes: ${NODES[*]}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SUBMIT_DIR=${SUBMIT_DIR}"
echo "WORKDIR=${WORKDIR}"
echo "LOCAL_NODE=${LOCAL_NODE}"
echo "LOCAL_HOST=${LOCAL_HOST}"
echo "DIST_SOCKET_IFNAME=${DIST_SOCKET_IFNAME:-<unset>}"
echo "TRAIN=${TRAIN_CMD}"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for node_rank in "${!NODES[@]}"; do
  node=${NODES[${node_rank}]}
  interface_exports=""
  if [[ -n "${DIST_SOCKET_IFNAME}" ]]; then
    interface_exports="
    export GLOO_SOCKET_IFNAME=$(printf "%q" "${DIST_SOCKET_IFNAME}")
    export NCCL_SOCKET_IFNAME=$(printf "%q" "${DIST_SOCKET_IFNAME}")
    export UCX_NET_DEVICES=$(printf "%q" "${DIST_SOCKET_IFNAME}")
    "
  fi

  remote_command="
    set -euo pipefail
    ${REMOTE_PREFIX}
    cd $(printf "%q" "${WORKDIR}")
    export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0,1}
    export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=\${MKL_NUM_THREADS:-1}
    ${interface_exports}
    echo \"[\$(hostname)] node_rank=${node_rank} starting torchrun\"
    echo \"[\$(hostname)] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}\"
    echo \"[\$(hostname)] GLOO_SOCKET_IFNAME=\${GLOO_SOCKET_IFNAME:-<unset>}\"
    torchrun \
      --nnodes=${NNODES} \
      --nproc_per_node=${GPUS_PER_NODE} \
      --node_rank=${node_rank} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      ${TRAIN_CMD}
  "

  if is_local_node "${node}"; then
    bash -lc "${remote_command}" &
  else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -- "${node}" \
      "bash -lc $(printf "%q" "${remote_command}")" &
  fi
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

trap - INT TERM
exit "${status}"
