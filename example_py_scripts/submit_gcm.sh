#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 TRAIN_SCRIPT [TRAIN_ARGS...]"
  echo
  echo "Example:"
  echo "  $0 run_jupiter_moist.py -c jupiter_gcm_H2O-NH3_F100.yaml --output-dir output_6gpu"
  echo
  echo "Optional environment variables:"
  echo "  PADDLE_NODES='dart1 dart2 dart3'"
  echo "  GPUS_PER_NODE=2"
  echo "  MASTER_ADDR=dart1"
  echo "  MASTER_PORT=29500"
  echo "  WORKDIR=$(pwd)  # defaults to the directory where this launcher is submitted"
  exit 2
fi

read -r -a NODES <<< "${PADDLE_NODES:-dart2 dart3 dart1}"
NNODES=${#NODES[@]}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
MASTER_ADDR=${MASTER_ADDR:-${NODES[0]}}
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-paddle-$(date +%Y%m%d%H%M%S)-$$}
SUBMIT_DIR=$(pwd)
WORKDIR=${WORKDIR:-${SUBMIT_DIR}}
LOCAL_HOST=$(hostname)
LOCAL_SHORT_HOST=$(hostname -s)

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

TRAIN_CMD="$(quote_args "${TRAIN_SCRIPT}" "$@")"
REMOTE_PREFIX='if [[ -f "${HOME}/.bash_profile" ]]; then source "${HOME}/.bash_profile"; fi'

echo "Launching ${NNODES} nodes x ${GPUS_PER_NODE} GPUs = $((NNODES * GPUS_PER_NODE)) processes"
echo "Nodes: ${NODES[*]}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RDZV_ID=${RDZV_ID}"
echo "SUBMIT_DIR=${SUBMIT_DIR}"
echo "WORKDIR=${WORKDIR}"
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
  remote_command="
    set -euo pipefail
    ${REMOTE_PREFIX}
    cd $(printf "%q" "${WORKDIR}")
    export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0,1}
    export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=\${MKL_NUM_THREADS:-1}
    echo \"[\$(hostname)] node_rank=${node_rank} starting torchrun\"
    torchrun \
      --nnodes=${NNODES} \
      --nproc_per_node=${GPUS_PER_NODE} \
      --node_rank=${node_rank} \
      --rdzv_backend=c10d \
      --rdzv_id=$(printf "%q" "${RDZV_ID}") \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      ${TRAIN_CMD}
  "

  if [[ "${node}" == "${LOCAL_HOST}" || "${node}" == "${LOCAL_SHORT_HOST}" ]]; then
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
