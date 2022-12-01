python src/evaluate.py \
--weights checkpoints/train-batch_all-resnet101-bsz1-lr3e3-decay1e-4/checkpoints/last.ckpt \
--backbone resnet101 \
--base_lr 3e-3 \
--batch_size 1 \
--max_epochs 30 \
--top_k 10
