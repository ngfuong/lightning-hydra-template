python src/train_online.py \
--exp-name train-online-batch_all-resnet50-bsz1-lr3e5-decay1e5 --project_name visual-search \
--backbone resnet50 \
--loss batch_all \
--batch_size 1 --base_lr 3e-5 \
--weight_decay 1e-5 \
--momentum 0.9 