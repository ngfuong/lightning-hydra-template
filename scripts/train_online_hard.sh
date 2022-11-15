#!/bin/bash
python src/train_online.py \
--exp-name train-online-batch_hard-resnet50-bsz1-lr1e3-decay1e-4 --project_name visual-search \
--backbone resnet50 \
--loss batch_hard \
--batch_size 1 --base_lr 1e-3 \
--weight_decay 1e-4 \
--momentum 0.9