from argparse import ArgumentParser
import os
from xmlrpc.client import boolean
import glob

import torch
import torch.optim as optim
import torch.nn as nn

from src.models.vs_module import VisualSearchModule
from src.utils.logger import Logger
from src.utils import utils

import pytorch_lightning as pl

def do_training(hparams, model_constructor):
    # Instantiate model
    model = model_constructor(**vars(hparams))

    # Set training params
    hparams.gpus = -1
    hparams.accelerator = 'ddp'
    hparams.benchmark = True

    if hparams.resume:
        hparams = set_resume_parameters(hparams)

    # Loggers
    wblogger = utils.get_wandb_logger(hparams)
    hparams.logger = [wblogger]

    hparams.callbacks = make_checkpoint_callbacks(hparams.exp_name)
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model)
    

def train(epoch, model, dataloader, optimizer, training):
    utils.fix_randseed()
    model.module.train_mode() if training else model.module.eval()

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda()
        logits = model.forward(batch['images'], batch['ids'])
        preds = logits.argmax(dim=1)

        loss = model.module.compute_loss(logits, batch['ids'])
        

def get_wandb_logger(hparams):
    exp_dir = f"checkpoints/{hparams.exp_name}/"
    id_file = f"{exp_dir}/wandb_id"

    if os.path.exists(id_file):
        with open(id_file) as f:
            hparams.wandb_id = f.read()
    else:
        hparams.wandb_id = None
    
    if hparams.wandb_id is None:
        _ = logger.experiment
    
    logger = pl.loggers.WandbLogger(
        project="visual-search",
        save_dir="checkpoints",
        name=hparams.exp_name,
        id=hparams.wandb_id
    )

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

        with open(id_file, 'w') as f:
            f.write(logger.version)
    
    return logger

def get_default_argument_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        '--project-name',
        type=str,
        default='visual-search',
    )

    parser.add_argument(
        '--exp-name',
        type=str,
        required=True
    )

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
    )

    parser.add_argument(
        '--num_nodes',
        type=int,
        default=1,
        help="number of nodes for distributed training"
    )

    parser.add_argument(
        '--resume',
        action="store_false",
        default=False,
        help="resume if have a checkpoint"
    )

    return parser

def make_checkpoint_callbacks(exp_name, version, base_path='checkpoints', frequency=1):
    base_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{base_path}/{exp_name}/checkpoints",
        save_last=True,
        verbose=True
    )

    val_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_best",
        dirpath=f"{base_path}/{exp_name}/checkpoints",
        filename="result-{epoch}-{val_loss:.2f}",
        mode="max",
        save_top_k=-1,
        verbose=True,
    )

    return [base_callback, val_callback]

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


