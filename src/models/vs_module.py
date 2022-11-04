from typing import Any, List
import math
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision.models import resnet50
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

# import sklearn

from utils.logger import Logger
from datamodules.datasets import VSDataset

class VS_args:
    datapath = 'data/deepfashion2'
    benchmark = "deepfashion"
    logpath = 'logs'
    nworker = 8
    bsz = 16


class VisualSearchModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        datapath: str,
        dataset: str,
        batch_size: int,
        base_lr,
        max_epochs,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['kwargs'])

        # self.benchmark = dataset
        self.batch_size = batch_size    # base_batch_size = 16
        self.base_lr = base_lr * math.sqrt(self.batch_size / 16)
        self.epochs = max_epochs
        
        self.args = self.get_args()
        self.args.datapath = datapath
        self.args.benchmark = dataset
        self.args.bsz = self.batch_size
        self.other_kwargs = kwargs
        self.write_batch_idx = 1000 * 16 / self.batch_size

        self.criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_loss_best = MinMetric()

        # TODO: ADD MORE MODEL
        if kwargs['backbone'] in ["resnet50"]:
            VSDataset.initialize(img_size=224, datapath=self.args.datapath,imagenet_norm=True)
            self.net = resnet50(pretrained=True)
            num_features = self.net.fc.in_features
            self.net.fc = nn.Linear(num_features, 128) # 128 = num_classes
        else:
            VSDataset.initialize(img_size=224, datapath=self.args.datapath)
            self.net = None
        
        self.best_val_loss = float('-inf')

        Logger.initialize(self.args, training=True)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def get_args(self):
        return VS_args()

    """
    def step(self, batch: Any):
        images = batch
        anchor, positive, negative = images[0], images[1], images[2]
        anchor, positive, negative = self.forward(anchor), self.forward(positive), self.forward(negative)
        loss = self.criterion(anchor, positive, negative)
        #TODO: compute logits (to compare with labels y)
        # preds_positive = torch.argmax(positive, dim=1)
        # preds_negative = torch.argmax(negative, dim=1)
        # preds = anchor, preds_positive, preds_negative
        # return loss, preds, y
        # return loss, y
        return loss
    """

    def training_step(self, batch: Any, batch_idx: int):
        # loss, preds, targets = self.step(batch)

        images, ids = batch

        a, p, n = [self(x) for x in images]
        loss = self.criterion(a, p, n)
        
        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss.update(loss)
        # self.train_acc(preds, targets)
        if self.global_rank == 0:
            if batch_idx % self.write_batch_idx == 0: 
                msg = '[Epoch: %02d] ' % self.current_epoch
                msg += '[Batch: %06d/%06d] ' % (batch_idx+1, self.len_train_dataloader)
                msg += 'L: %6.5f ' % loss
                msg += 'Avg L: %6.5f ' % self.train_loss.compute()
                Logger.info(msg)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        # return {"loss": loss, "preds": preds, "targets": targets}
        return loss
        # return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        if self.global_rank == 0:
            self.log('Training', self.current_epoch)

        msg = '\n*** Train '
        msg += '[@Epoch %02d] ' % self.current_epoch
        msg += 'Avg L: %6.5f' % self.train_loss.compute()
        msg += '***\n'
        # self.log("Train Avg L", self.train_loss.compute(), on_epoch=True)
        Logger.info(msg)


    def validation_step(self, batch: Any, batch_idx: int):
        images, ids = batch

        a, p, n = [self(x) for x in images]
        loss = self.criterion(a, p, n)

        # update and log metrics
        self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss.update(loss)
        # self.val_acc(preds, targets)
        if self.global_rank == 0:
            if batch_idx % self.write_batch_idx == 0: 
                msg = '[Epoch: %02d] ' % self.current_epoch
                msg += '[Batch: %06d/%06d] ' % (batch_idx+1, self.len_val_dataloader)
                msg += 'L: %6.5f' % loss
                msg += 'Avg L: %6.5f' % self.val_loss.compute()
                Logger.info(msg)

        # return {"loss": loss, "preds": preds, "targets": targets}
        return loss
        # return {"loss": loss, "ids": ids}

    def validation_epoch_end(self, outputs: List[Any]):
        if self.global_rank == 0:
            self.log('Validation', self.current_epoch)
            
        loss = self.val_loss.compute()  # get epoch val loss
        msg = '\n*** Validation'
        msg += '[@Epoch %02d] ' % self.current_epoch
        msg += 'Avg L: %6.5f' % loss
        msg += '***\n'
        Logger.info(msg)

        self.val_loss_best.update(loss)  # update best so far val loss
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

        if self.global_rank==0:
            Logger.tbd_writer.add_scalars('loss', {'train': self.train_loss.compute(), 'val': loss}, self.current_epoch)
            Logger.tbd_writer.flush()
            if self.current_epoch + 1 == self.epochs:
                Logger.tbd_writer.close()
                Logger.info('=========== Finished Training ===========')

    def test_step(self, batch: Any, batch_idx: int):
        """
        images,ids = batch

        a, p, n = [self(x) for x in images]
        loss = self.criterion(a, p, n)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "ids": ids}
        """
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # optimizer = self.hparams.optimizer(params=self.parameters())
        # scheduler = self.hparams.scheduler(optimizer=optimizer)
        params_list = [
            {"params": self.net.parameters(), "lr": self.base_lr},
        ]

        optimizer = torch.optim.SGD(
            params_list,
            lr=self.base_lr,
            momentum=self.other_kwargs['weight_decay'],
            weight_decay=self.other_kwargs['weight_decay']
        )
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: pow(1.0 -x / self.epochs, 0.9)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def train_dataloader(self):
        dataloader = VSDataset.build_dataloader(
            self.args.benchmark,
            self.args.bsz,
            self.args.nworker,
            split='train',
        )
        
        self.len_train_dataloader = len(dataloader)//torch.cuda.device_count()
        return dataloader

    def val_dataloader(self):
        dataloader = VSDataset.build_dataloader(
            self.args.benchmark,
            self.args.bsz,
            self.args.nworker,
            split='val',
        )
        
        self.len_val_dataloader = len(dataloader)//torch.cuda.device_count()
        return dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--datapath",
            type=str,
            default='data/deepfashion2',
            help='path of datasets'
        )

        parser.add_argument(
            "--dataset",
            type=str,
            default='deepfashion',
            choices=['deepfashion']
        )

        parser.add_argument(
            '--batch_size',
            type=int,
            default=16
        )

        parser.add_argument(
            '--base_lr',
            type=float,
            default=1e-3
        )

        parser.add_argument(
            '--weight_decay',
            type=float,
            default=1e-4
        )
        
        parser.add_argument(
            '--momentum',
            type=float,
            default=0.9
        )

        return parser
    

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
