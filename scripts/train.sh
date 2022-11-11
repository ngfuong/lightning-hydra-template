#!/bin/bash
python src/train.py \
--exp-name train-v1-bsz16-lr1e3-small --project_name visual-search \
--backbone resnet50 \
--batch_size 16 --base_lr 1e-3 \
--weight_decay 1e-4 \
--momentum 0.9

