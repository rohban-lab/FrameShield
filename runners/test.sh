#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --rdzv_endpoint=localhost:29450 --nproc_per_node=$1 main.py -cfg configs/traffic/traffic_server.yaml --output output/test --pretrained ../weights/best.pth --only_test