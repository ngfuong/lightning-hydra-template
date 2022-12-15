python src/train_online.py \
--exp-name train-batch_all-resnet50-bsz1-lr3e3-decay1e4_demo --project_name visual-search \
--datapath ~/lightning-hydra-template/data \
--backbone resnet50 \
--loss batch_all \
--batch_size 1 --base_lr 3e-3 \
--weight_decay 1e-4 \
--momentum 0.9 
