#!/usr/bin/env bash
set -ex

echo "Working Directory = $(pwd)"
echo "ID=$(id)"

source /opt/venv/bin/activate
#sleep 600

# Run unbuffered, capture stdout+stderr
#echo "$0" > a.log
#echo "$1" >> a.log
#echo "$2" >> a.log
#python -u train.py >> a.log 2>&1

torchrun \
    --nnodes=$4 \
    --nproc_per_node=$3 \
    --node_rank=$2 \
    --master_addr=$1 \
    --master_port=29500 \
    train.py > a.log 2>&1

echo "DONE"
sleep 10
