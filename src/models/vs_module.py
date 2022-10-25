from typing import Any, List
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

# import sklearn

from src.utils.logger import Logger
from src.datamodules.datasets import VSDataset


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

        self.args.datapath = datapath
        self.benchmark = dataset
        self.bsz = batch_size
        self.base_lr = base_lr
        self.epochs = max_epochs

        self.other_kwargs = kwargs

        self.criterion = torch.nn.TripletMarginLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_loss_best = MaxMetric()

        if kwargs['backbone'] in ["resnet50"]:
            VSDataset.initialize(img_size=224, datapath=self.args.datapath,imagenet_norm=True)
        else:

            VSDataset.initialize(img_size=224, datapath=self.args.datapath)
        
        self.best_val_loss = float('-inf')
        self.train_loss = 100

        Logger.initialize()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        images = batch['images']
        anchor, positive, negative = images[0], images[1], images[2]
        anchor, positive, negative = self.forward(anchor), self.forward(positive), self.forward(negative)
        loss = self.criterion(anchor, positive, negative, margin=1.0, p=2)
        #TODO: compute logits (to compare with labels y)
        # preds_positive = torch.argmax(positive, dim=1)
        # preds_negative = torch.argmax(negative, dim=1)
        # preds = anchor, preds_positive, preds_negative
        # return loss, preds, y
        # return loss, y
        return loss, y

    def training_step(self, batch: Any, batch_idx: int):
        # loss, preds, targets = self.step(batch)
        loss, targets = self.step(batch)

        # update and log metrics
        self.train_loss.update(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        # return {"loss": loss, "preds": preds, "targets": targets}
        return {"loss": loss, "targets": targets}
        # return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        if self.global_rank == 0:
            self.log('Traning', self.current_epoch)
        self.log("AvgTrainLoss", self.train_loss.compute(), on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, targets = self.step(batch)

        # update and log metrics
        self.val_loss.update(loss)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return {"loss": loss, "preds": preds, "targets": targets}
        return {"loss": loss, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        if self.global_rank == 0:
            self.log('Validation', self.current_epoch)
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

        if self.global_rank==0:
            Logger.tbd_writer.add_scalar('train_loss', {self.train_loss}, self.current_epoch)
            Logger.tbd_writer.add_scalars('val_loss', {self.val_loss}, self.current_epoch)
            Logger.tbd_writer.flush()
            if self.current_epoch + 1 == self.epochs:
                Logger.tbd_writer.close()
                Logger.info('=========== Finished Training ===========')

    def test_step(self, batch: Any, batch_idx: int):
        loss, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "targets": targets}

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
            momentum=0.9,
            weight_decay=1e-3
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
            'train',
        )
        
        self.len_train_dataloader = len(dataloader)//torch.cuda.device_count()
        return dataloader

    def val_dataloader(self):
        dataloader = VSDataset.build_dataloader(
            self.args.benchmark,
            self.args.bsz,
            self.args.nworker,
            'val',
        )
        
        self.len_val_dataloader = len(dataloader)//torch.cuda.device_count()
        return dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--datapath"
        )
    

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
