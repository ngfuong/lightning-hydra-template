from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

# from src.datamodules.datasets import DeepFashionOnlineTripletDataset
from src.datamodules.datasets import DeepFashionOnlineTripletBalanceDataset


class OnlineTripletDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        benchmark: str = "deepfashion",
        data_dir: str = "data/",
        img_size: int = 256,
        imagenet_norm: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.datasets = {
            "deepfashion": DeepFashionOnlineTripletBalanceDataset,
        }

        if imagenet_norm:
            self.img_mean = [0.485, 0.456, 0.406]
            self.img_std = [0.229, 0.224, 0.225]
        else:
            self.img_mean = [0.5, 0.5, 0.5]
            self.img_std = [0.5, 0.5, 0.5]
        print(f"Use norm img_mean {self.img_mean}\timg_std{self.img_std}")

        self.benchmark = benchmark
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # self.data_train: Optional[Dataset] = None
        # self.data_val: Optional[Dataset] = None
        # self.data_test: Optional[Dataset] = None

    @classmethod
    def build_dataloader(self, benchmark, bsz, nworker, split, pin_memory=False):
        nworker = nworker
        if split == "train":
            shuffle = True
            transforms = transforms.Compose(
                [
                    transforms.Resize(size=(self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(self.img_mean, self.img_std),
                ]
            )
        else:
            transforms = transforms.Compose(
                [
                    transforms.Resize(size=(self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.img_mean, self.img_std),
                ]
            )

        dataset = self.datasets[benchmark](
            self.datapath, transforms=transforms, split=split
        )
        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=shuffle,
            num_workers=nworker,
            pin_memory=True,
        )

        return dataloader

    @property
    def num_classes(self):
        return 13

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        pass
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = self.datasets[self.benchmark](self.data_dir, transforms=self.transforms, split='train')
            valset = self.datasets[self.benchmark](self.data_dir, transforms=self.transforms, split='val')
            dataset = ConcatDataset(datasets=[trainset, valset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
        """

    def train_dataloader(self):
        data_train = self.build_dataloader(
            benchmark=self.benchmark,
            bsz=self.batch_size,
            nworker=self.num_workers,
            split="train",
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        self.data_val = self.build_dataloader(
            benchmark=self.benchmark,
            bsz=self.batch_size,
            nworker=self.num_workers,
            split="val",
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
