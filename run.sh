#!/usr/bin/env bash
set -ex

echo "Working Directory = $(pwd)"
echo "ID=$(id)"

source /opt/venv/bin/activate
#sleep 600

MASTER_PORT=29501 # \
MASTER_ADDR=$1 # \
NODE_RANK=$2 # \
PROC_PER_NODE=$3 # \
NODES=$4 # \
#WORLD_SIZE=$4 # \
#CUDA_VISIBLE_DEVICES="0,1" \
#python /work/straka.py

torchrun \
    --nnodes=$NODES \
    --nproc_per_node=$PROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /work/straka.py > a.log 2>&1

echo "DONE"
#sleep 10
