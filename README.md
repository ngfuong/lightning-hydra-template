# Setup
## Expected data structure
Root dataset is default to `data/`
```
data/
    train/                      # downloaded deepfashion2 train set
    val/                        # downloaded deepfashion2 validation set
    val_gallery.json            # generated from df2
    val_query.json              # generated from df2
    list_bbox_consumer2shop.txt # downloaded for deepfashion2 evaluation
    deepfashion/                # downloaded deepfashion set
        img/
        triplets.csv            # generated triplets from df
        train_triplets.csv      # split randomly from triplets.csv
        val_triplets.csv        # split randomly from triplets.csv
    shopping100k/               # downloaded shopping100k dataset
    shopping100k_similar.csv    # generated for shopping100k evaluation
```

## Create train/val/test environment
In root (repo) directory, create and activate environment using `pip`:
```
python -m venv ./env
source ./env/bin/activate
pip install -r requirements.txt
pip install -e .
```
**Optional:**
Before running train scripts, [authorize WandB](https://wandb.ai/authorize).
```
wandb login
<COPY AND ENTER API KEY>
```
# Train Scripts
Scripts are located under `scripts/`
Edit your arguments and hyperparameters in each `.sh` file.

## Train with script file
Make executable and run in shell:
```
chmod +x ./scripts/train_online.sh
./train_online.sh
```
## Train ResNet101 model with online sampling all samples
```
python src/train_online.py \
--exp-name train-batch_all-resnet101-bsz1-lr4e3-decay1e4 --project_name visual-search \
--datapath $PATH_TO_DEEPFASHION2_DATASET \ # default to ./data
--backbone resnet101 \
--loss batch_all \
--batch_size 1 --base_lr 4e-3 \
--weight_decay 1e-4 \
--momentum 0.9 
```
## Train ResNet50 with offline sampling
```
python src/train_offline.py \
--exp-name train-offline-resnet50-bsz64-lr3e3-decay1e-4 --project_name visual-search \
--datapath $PATH_TO_DEEPFASHION_DATASET --dataset deepfashion \
--backbone resnet50 \
--batch_size 64 --base_lr 3e-3 \
--weight_decay 1e-4 \
--momentum 0.9 \
```
# Test
## Evaluation on DeepFashion2
```
python src/evaluate_deepfashion2.py \
--df_path data/list_bbox_consumer2shop.txt \
--emb checkpoints/resnet101/top_k_acc=0.715.ckpt \
--save_dir checkpoints/embeddings \
```
## Evaluation on Shopping100k
```
python src/evaluate_shopping100k.py \
--attr_path data/shopping100k_similar.csv \
--emb_path checkpoints/embeddings/data_embeddings.npy \
--top 31 \
```


# Demo
Setup environment for visualization and access web application at `localhost:8000`
```
cd demo
pip install -r requirements.txt
python manage.py runserver
```

