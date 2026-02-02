#!/usr/bin/env bash
set -ex

echo "Working Directory = $(pwd)"
echo "ID=$(id)"

source /opt/venv/bin/activate

# Run unbuffered, capture stdout+stderr
#echo "AA"
python -u train.py > a.log 2>&1

echo "DONE"
sleep 10
