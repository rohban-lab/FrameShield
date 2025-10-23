#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --rdzv_endpoint=localhost:29450 main_attack.py -cfg configs/traffic/traffic_attack.yaml --batch-size 1 --accumulation-steps 8 --output output/attack --pretrained ../weights/best.pth