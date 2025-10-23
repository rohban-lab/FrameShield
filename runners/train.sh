#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --rdzv_endpoint=localhost:29450 --nproc_per_node=$1 main.py -cfg configs/traffic/traffic_server.yaml --batch-size 1 --accumulation-steps 8 --output output/train --pretrained ../weights/k400_16_8.pth