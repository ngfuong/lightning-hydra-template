# Setup
## Expected data structure for DeepFashion2
Root dataset is default to `data/`
```
data/
    deepfashion2/
        train/
        val/
        train_triplets.csv
        val_triplets.csv
```
## Create environment
In root (repo) directory, create and activate environment using `pip`:
```
python -m venv ./vs
source ./vs/bin/activate
pip install -r requirements.txt
pip install -e .
```
# Train Scripts
Scripts are located under `scripts/`
Edit your arguments and hyperparameters in each `.sh` file.

Before training, login with wandb to monitor logs.
Enter your API key at [Authorize Page](https://wandb.ai/authorize)
```
wandb login
<COPY AND ENTER API KEY>
```
## Train with script file
Make file executable and run in shell
```
chmod +x ./scripts/train.sh
./train.sh
```
## Train directly
```
python src/train.py \
--project-name visual-search --exp-name train_v1_lr1e-3_bsz1_e20 \
--backbone resnet50 
```
