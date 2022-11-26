import glob
import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from src.models.online_triplet_module import OnlineTripletModule
from src.utils import utils


def do_training(hparams, model_constructor):
    # Instantiate model
    model = model_constructor(**vars(hparams))

    # Set training params
    # hparams.gpus = -1
    hparams.accelerator = "cuda"
    hparams.benchmark = True

    if hparams.resume:
        hparams = set_resume_parameters(hparams)

    # Loggers
    wblogger = get_wandb_logger(hparams)
    hparams.logger = [wblogger]

    hparams.callbacks = make_checkpoint_callbacks(hparams.exp_name)
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model)


# TODO: write this
def train(epoch, model, dataloader, optimizer, training):
    utils.fix_randseed()
    model.module.train_mode() if training else model.module.eval()

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda()
        logits = model.forward(batch["images"], batch["ids"])
        preds = logits.argmax(dim=1)

        loss = model.module.compute_loss(logits, batch["ids"])


def get_wandb_logger(hparams):
    exp_dir = f"checkpoints/{hparams.exp_name}/"
    id_file = f"{exp_dir}/wandb_id"

    if os.path.exists(id_file):
        with open(id_file) as f:
            hparams.wandb_id = f.read()
    else:
        hparams.wandb_id = None

    logger = pl.loggers.WandbLogger(
        project=hparams.project_name,
        save_dir="checkpoints",
        name=hparams.exp_name,
        id=hparams.wandb_id,
    )

    if hparams.wandb_id is None:
        _ = logger.experiment

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

        with open(id_file, "w") as f:
            f.write(logger.version)

    return logger


def get_default_argument_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="visual-search",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--resume",
        action="store_false",
        default=False,
        help="resume if have a checkpoint",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="batch_hard",
        help="online triplet loss type",
    )

    return parser


def make_checkpoint_callbacks(exp_name, base_path="checkpoints", frequency=None):
    base_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{base_path}/{exp_name}/checkpoints", save_last=True, verbose=True
    )

    val_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        dirpath=f"{base_path}/{exp_name}/checkpoints/",
        filename="result-{epoch}-{val_loss_epoch:.2f}",
        mode="min",
        save_top_k=-1,  # save all checkpoints
        verbose=True,
    )

    train_earlystop = pl.callbacks.EarlyStopping(
        monitor="train/loss_epoch",
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode='min',
    )

    val_earlystop = pl.callbacks.EarlyStopping(
        monitor="val/loss_epoch",
        min_delta=0.005,
        patience=3,
        verbose=True,
        mode='min',
    )

    return [base_callback, val_callback, train_earlystop, val_earlystop]


def set_resume_parameters(hparams):
    latest = get_latest_checkpoint(hparams.exp_name)
    print(f"Resume checkpoint {latest}")
    hparams.resume_from_checkpoint = latest

    wandb_file = "checkpoints/{hparams.exp_name}/wandb_id"
    if os.path.exists(wandb_file):
        with open(wandb_file, "r") as f:
            hparams.wandb_id = f.read()

    return hparams


def get_latest_checkpoint(exp_name):
    folder = f"./checkpoints/{exp_name}/checkpoints/"
    latest = f"{folder}/last.ckpt"

    if os.path.exists(latest):
        return latest

    ckpts = glob(f"{folder}/epoch=*.ckpt")

    if len(ckpts) == 0:
        return None

    latest = max(ckpts, key=os.path.getctime)
    return latest


if __name__ == "__main__":
    parser = OnlineTripletModule.add_model_specific_args(get_default_argument_parser())
    hparams = parser.parse_args()
    print(hparams)
    do_training(hparams, OnlineTripletModule)
