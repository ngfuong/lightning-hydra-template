#!/bin/bash
export PYTHONPATH=.

python src/train_online.py \
--exp-name train-resnet50-bsz128-lr1e3 --project_name visual-search \
--backbone resnet50 \
--batch_size 128 --base_lr 1e-3 \
--weight_decay 1e-4 \
--momentum 0.9 
