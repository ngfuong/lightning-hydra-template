import math
from argparse import ArgumentParser
from typing import Any, List

import torch
import torch.nn as nn
from online_triplet_loss.losses import *
from pytorch_lightning import LightningModule
from sklearn.neighbors import NearestNeighbors
from torchmetrics import MeanMetric, MinMetric
from torchvision.models import resnet50

from datamodules.datasets import OnlineTripletDataset
from utils.logger import Logger
from utils.metrics import MeanReciprocalRank, TopKAccuracy
from utils.triplet_semi_hard_loss import TripletSemihardLoss

# import sklearn


class VS_args:
    datapath = "data"
    benchmark = "deepfashion"
    logpath = "logs"
    nworker = 1
    bsz = 1024


class OnlineTripletModule(LightningModule):
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
        self.save_hyperparameters(logger=False, ignore=["kwargs"])

        # self.benchmark = dataset
        self.batch_size = batch_size  # base_batch_size = 16
        self.base_lr = base_lr * math.sqrt(self.batch_size / 16)
        self.epochs = max_epochs

        self.args = self.get_args()
        self.args.datapath = datapath
        self.args.benchmark = dataset
        self.args.bsz = self.batch_size
        self.other_kwargs = kwargs
        self.write_batch_idx = 1000 * 16 / self.batch_size

        # define loss
        self.loss_type = kwargs.get("loss", "batch_all")
        if self.loss_type == "batch_all":
            self.criterion = batch_all_triplet_loss
        elif kwargs.get("loss") == "batch_hard":
            self.criterion = batch_hard_triplet_loss
        elif kwargs.get("loss") == "batch_semi_hard":
            self.criterion = TripletSemihardLoss
        else:
            self.criterion = TripletSemihardLoss

        # Top K Accuracy
        self.top_k = kwargs.get("top_k", 20)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_loss_best = MinMetric()
        self.top_k_accuracy = TopKAccuracy()
        self.mean_reciprocal_rank = MeanReciprocalRank()

        # TODO: ADD MORE MODEL
        if kwargs["backbone"] in ["resnet50"]:
            OnlineTripletDataset.initialize(
                img_size=224, datapath=self.args.datapath, imagenet_norm=True
            )
            self.net = resnet50(pretrained=True)
            num_features = self.net.fc.in_features
            self.net.fc = nn.Linear(num_features, 128)  # 128 = num_classes
        else:
            OnlineTripletDataset.initialize(img_size=224, datapath=self.args.datapath)
            self.net = None

        self.best_val_loss = float("-inf")

        Logger.initialize(self.args, training=True)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def get_args(self):
        return VS_args()

    def training_step(self, batch: Any, batch_idx: int):
        images, ids = batch

        images = torch.squeeze(torch.cat(images, 0), 0)
        ids = torch.squeeze(torch.cat(ids, 0), 0)

        embeddings = self(images)

        if self.loss_type == "batch_all":
            loss, _ = self.criterion(
                labels=ids, embeddings=embeddings, margin=1, squared=True
            )
        elif self.loss_type == "batch_hard":
            loss = self.criterion(
                labels=ids, embeddings=embeddings, margin=1, squared=True
            )
        elif self.loss_type == "batch_semi_hard":
            loss = self.criterion(
                targets=ids, embeddings=embeddings, margin=1, squared=True
            )

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss.update(loss)
        # self.train_acc(preds, targets)
        if self.global_rank == 0:
            if batch_idx % self.write_batch_idx == 0:
                msg = "[Epoch: %02d] " % self.current_epoch
                msg += "[Batch: %06d/%06d] " % (
                    batch_idx + 1,
                    self.len_train_dataloader,
                )
                msg += "L: %6.5f " % loss
                msg += "Avg L: %6.5f " % self.train_loss.compute()
                Logger.info(msg)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        # return {"loss": loss, "preds": preds, "targets": targets}
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        if self.global_rank == 0:
            loss = self.train_loss.compute()
            msg = "\n*** Train "
            msg += "[@Epoch %02d] " % self.current_epoch
            msg += "Avg L: %6.5f" % loss
            msg += "***\n"
            Logger.info(msg)

        # Log epoch loss
        self.log("train/loss_epoch", loss)

    def on_validation_start(self):
        gallery_dataloader = self.val_dataloader("gallery")

        gallery_vectors = []
        gallery_classes = []
        class_ids = {}
        class_count = 0

        for batch_idx, batch in enumerate(gallery_dataloader):
            imgs, pair_ids, styles = batch
            imgs = torch.stack(imgs, 0).to("cuda")

            feature_vector = self(imgs)
            gallery_vectors.append(feature_vector.to("cpu"))
            # label
            for i in range(len(pair_ids)):
                label = f"{pair_ids[i]}_{styles[i]}"
                if label not in class_ids:
                    class_ids[f"{pair_ids[i]}_{styles[i]}"] = class_count
                    class_count += 1
                gallery_classes.append(class_ids[label])

        gallery_vectors = torch.cat(gallery_vectors, 0)
        self.knn = NearestNeighbors(n_neighbors=20, n_jobs=-1)
        self.knn.fit(gallery_vectors)

        self.class_ids = class_ids
        self.gallery_classes = torch.Tensor(gallery_classes)

    def validation_step(self, batch: Any, batch_idx: int):
        imgs, pair_ids, styles = batch
        imgs = torch.stack(imgs, 0).to("cuda")
        embeddings = self(imgs)

        # Convert label
        labels = []
        assert len(pair_ids) == len(styles)
        for i in range(len(pair_ids)):
            label = self.class_ids[f"{pair_ids[i]}_{styles[i]}"]
            labels.append(label)

        dists, indexes = self.knn.kneighbors(embeddings.to("cpu"), self.top_k)
        top_k_classes = self.gallery_classes[indexes]
        # calculate top k acc
        self.top_k_accuracy.update(labels, top_k_classes)
        #  mean reciprocal rank
        self.mean_reciprocal_rank.update(labels, top_k_classes)

        # if self.loss_type == "batch_all":
        #     loss, _ = self.criterion(
        #         labels=ids, embeddings=embeddings, margin=1, squared=True
        #     )
        # elif self.loss_type == "batch_hard":
        #     loss = self.criterion(
        #         labels=ids, embeddings=embeddings, margin=1, squared=True
        #     )

        # # update and log metrics
        # self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        # # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.val_loss.update(loss)
        # # self.val_acc(preds, targets)
        # if self.global_rank == 0:
        #     if batch_idx % self.write_batch_idx == 0:
        #         msg = "[Epoch: %02d] " % self.current_epoch
        #         msg += "[Batch: %06d/%06d] " % (batch_idx + 1, self.len_val_dataloader)
        #         msg += "L: %6.5f" % loss
        #         msg += "Avg L: %6.5f" % self.val_loss.compute()
        #         Logger.info(msg)

        # return loss

    def validation_epoch_end(self, outputs: List[Any]):
        top_k_acc = self.top_k_accuracy.compute()
        mrr = self.mean_reciprocal_rank.compute()
        self.log("val/top_k_acc", top_k_acc, on_epoch=True)
        self.log("val/mean_reciprocal_rank", mrr, on_epoch=True)
        # if self.global_rank == 0:
        #     loss = self.val_loss.compute()  # get epoch val loss
        #     msg = "\n*** Validation"
        #     msg += "[@Epoch %02d] " % self.current_epoch
        #     msg += "Avg L: %6.5f" % loss
        #     msg += "***\n"
        #     Logger.info(msg)

        # self.log("val/loss_epoch", loss, on_epoch=True)
        # self.val_loss_best.update(loss)  # update best so far val loss
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

        if self.global_rank == 0:
            Logger.tbd_writer.add_scalars(
                "loss",
                {
                    "train": self.train_loss.compute(),
                    "val/top_k_acc": top_k_acc,
                    "val/mean_reciprocal_rank": mrr,
                },
                self.current_epoch,
            )
            Logger.tbd_writer.flush()
            if self.current_epoch + 1 == self.epochs:
                Logger.tbd_writer.close()
                Logger.info("=========== Finished Training ===========")

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
            momentum=self.other_kwargs["weight_decay"],
            weight_decay=self.other_kwargs["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: pow(1.0 - x / self.epochs, 0.9)
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
        dataloader = OnlineTripletDataset.build_dataloader(
            self.args.benchmark,
            self.args.bsz,
            self.args.nworker,
            split="train",
        )

        self.len_train_dataloader = len(dataloader) // torch.cuda.device_count()
        return dataloader

    def val_dataloader(self, val_type="query"):
        dataloader = OnlineTripletDataset.build_dataloader(
            self.args.benchmark, 1, self.args.nworker, "val", val_type
        )

        self.len_val_dataloader = len(dataloader) // torch.cuda.device_count()
        return dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--datapath", type=str, default="data", help="path of datasets"
        )

        parser.add_argument(
            "--dataset", type=str, default="deepfashion", choices=["deepfashion"]
        )

        parser.add_argument("--batch_size", type=int, default=16)

        parser.add_argument("--base_lr", type=float, default=1e-3)

        parser.add_argument("--weight_decay", type=float, default=1e-4)

        parser.add_argument("--momentum", type=float, default=0.9)

        return parser


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
