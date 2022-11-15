python src/train_online.py \
--exp-name train-online-batch_all-resnet50-bsz1-lr1e4-decay1e5 --project_name visual-search \
--backbone resnet50 \
--loss batch_all \
--batch_size 1 --base_lr 1e-4 \
--weight_decay 1e-5 \
--momentum 0.9 