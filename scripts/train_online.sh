python src/train_online.py \
--exp-name train-online-batch-hard_resnet50-bsz4-lr1e4 --project_name visual-search \
--backbone resnet50 \
--loss batch_hard \
--batch_size 4 --base_lr 1e-4 \
--weight_decay 1e-4 \
--momentum 0.9 
