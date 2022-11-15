import os
import random

import pandas as pd
import torch
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import DataLoader

from src.utils.utils import collate_fn, random_exclusion
from src.datamodules.deepfashion import DatasetDeepFashion
from src.datamodules.onlinetriplets import DeepFashionOnlineTripletBalanceDataset


class VSDataset:
    @classmethod
    def initialize(cls, img_size, datapath, imagenet_norm=False):
        cls.datasets = {
            "deepfashion": DatasetDeepFashion,
        }

        if imagenet_norm:
            cls.img_mean = [0.485, 0.456, 0.406]
            cls.img_std = [0.229, 0.224, 0.225]
        else:
            cls.img_mean = [0.5, 0.5, 0.5]
            cls.img_std = [0.5, 0.5, 0.5]
        print(f"Use norm img_mean {cls.img_mean}\timg_std {cls.img_std}")

        cls.img_size = img_size
        cls.datapath = datapath

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, split):
        nworker = nworker
        shuffle = False
        if split == "train":
            shuffle = True
            transforms = transform.Compose(
                [
                    transform.Resize(size=(cls.img_size, cls.img_size)),
                    transform.RandomHorizontalFlip(p=0.5),
                    transform.ToTensor(),
                    transform.Normalize(cls.img_mean, cls.img_std),
                ]
            )
        else:
            transforms = transform.Compose(
                [
                    transform.Resize(size=(cls.img_size, cls.img_size)),
                    transform.ToTensor(),
                    transform.Normalize(cls.img_mean, cls.img_std),
                ]
            )

        dataset = cls.datasets[benchmark](
            cls.datapath, transforms=transforms, split=split
        )
        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=shuffle,
            num_workers=nworker,
            pin_memory=True,
        )

        return dataloader


class OnlineTripletDataset:
    @classmethod
    def initialize(cls, img_size, datapath, imagenet_norm=False):
        cls.datasets = {
            "deepfashion": DeepFashionOnlineTripletBalanceDataset,
        }

        if imagenet_norm:
            cls.img_mean = [0.485, 0.456, 0.406]
            cls.img_std = [0.229, 0.224, 0.225]
        else:
            cls.img_mean = [0.5, 0.5, 0.5]
            cls.img_std = [0.5, 0.5, 0.5]
        print(f"Use norm img_mean {cls.img_mean}\timg_std {cls.img_std}")

        cls.img_size = img_size
        cls.datapath = datapath

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, split):
        nworker = nworker
        shuffle = False
        if split == "train":
            shuffle = True
            transforms = transform.Compose(
                [
                    transform.Resize(size=(cls.img_size, cls.img_size)),
                    transform.RandomHorizontalFlip(p=0.5),
                    transform.ToTensor(),
                    transform.Normalize(cls.img_mean, cls.img_std),
                ]
            )
        else:
            transforms = transform.Compose(
                [
                    transform.Resize(size=(cls.img_size, cls.img_size)),
                    transform.ToTensor(),
                    transform.Normalize(cls.img_mean, cls.img_std),
                ]
            )

        dataset = cls.datasets[benchmark](
            cls.datapath, transforms=transforms, split=split
        )
        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=shuffle,
            num_workers=nworker,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return dataloader
