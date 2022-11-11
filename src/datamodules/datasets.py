import os
import random

import pandas as pd
import torch
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.utils import random_exclusion


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
        nworker = nworker if split == "train" else 0
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


class DatasetDeepFashion(Dataset):
    def __init__(self, datapath, transforms, split):
        self.split = "val" if split in ["val", "test"] else "train"
        self.datapath = datapath
        self.img_path = os.path.join(self.datapath, self.split, "image")
        self.transforms = transforms
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata["image_name"])

    def __getitem__(self, idx):
        # TODO: check this if data per batch is not loaded correctly
        triplet_paths, triplet_boxes, triplet_ids = self.sample(idx)
        triplet_imgs = self.load_frames(triplet_paths, boxes=triplet_boxes)
        # print("after load frames")
        # len(triplet_imgs)= 3 ; type(triplet_imgs) = list
        # type(triplet_imgs[0])=PIL Image
        triplet_imgs = [self.transforms(imgs) for imgs in triplet_imgs]
        # print(triplet_imgs[0].shape=(3, 224, 224))

        # batch = {
        # 'images': triplet_imgs, # list(torch.tensor(3, 224, 224)*3)
        # 'ids': triplet_ids, # nd.array([np.int64, np.int64, np.int64])
        # }
        return triplet_imgs, triplet_ids

    def load_frames(self, paths, boxes=None):
        # load images from paths
        # if boxes, crop corresponding box (x1,y1,x2,y2) from image
        frames = []
        for path, box in zip(paths, boxes):
            frames.append(self.read_image(path, box=box))
        return frames

    def read_image(self, image_name, box=None):
        r"""Return RGB image in PIL Image"""
        image = Image.open(os.path.join(self.img_path, image_name))
        if box is not None:
            image = image.crop(box)
        return image.convert("RGB")

    def sample(self, idx):
        """Sample path of triplet of images from the dataset"""
        # a_path, p_path, n_path = self.img_metadata['image_name'].iloc[idx].values
        paths = self.img_metadata["image_name"].iloc[idx].values
        # a_box, p_box, n_box = [self.img_metadata['box'].iloc[idx].values for _ in range(3)]
        boxes = self.img_metadata["box"].iloc[idx].values.reshape((3, -1))
        ids = self.img_metadata["category_id"].iloc[idx].values
        return paths, boxes, ids

    def build_class_ids(self):
        """Build a dictionary of class ids"""
        class_ids = {}
        for i, class_name in enumerate(self.df["class_name"].unique()):
            class_ids[class_name] = i
        return class_ids

    def build_img_metadata(self):
        def read_metadata(df):
            """Return metadata
            metadata: Dictionary
                image_name
                box
                category_id
            """
            metadata = {
                "image_name": df[["image_name_a", "image_name_p", "image_name_n"]],
                "box": df[
                    [
                        "x_1_a",
                        "y_1_a",
                        "x_2_a",
                        "y_2_a",
                        "x_1_p",
                        "y_1_p",
                        "x_2_p",
                        "y_2_p",
                        "x_1_n",
                        "y_1_n",
                        "x_2_n",
                        "y_2_n",
                    ]
                ],
                "category_id": df[["category_id_a", "category_id_p", "category_id_n"]],
            }
            return metadata

        def load_csv(split):
            with open(os.path.join(self.datapath, split + "_triplets.csv"), "r") as f:
                df = pd.read_csv(f)
                return df

        df = load_csv(self.split)

        img_metadata = {}
        if self.split in ["train", "val"]:
            img_metadata.update(read_metadata(df))
        else:
            raise Exception("Undefined split %s: " % self.split)

        print(
            "Total (%s) images are: %d" % (self.split, len(img_metadata["image_name"]))
        )
        return img_metadata


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
        nworker = nworker if split == "train" else 0
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


class DeepFashionOnlineTripletDataset(Dataset):
    def __init__(self, datapath, transforms, split):
        self.split = "val" if split in ["val", "test"] else "train"
        self.datapath = datapath
        self.img_path = os.path.join(self.datapath, self.split, "image")
        self.transforms = transforms
        self.img_metadata = self.build_img_metadata()
        self.class_ids = self.build_class_ids()

    def __len__(self):
        return len(self.img_metadata["image_name"])

    def __getitem__(self, idx):
        # TODO: check this if data per batch is not loaded correctly
        path, boxes, label = self.sample(idx)
        img = self.load_frames(path, boxes=boxes)
        # print("after load frames")
        # len(triplet_imgs)= 3 ; type(triplet_imgs) = list
        # type(triplet_imgs[0])=PIL Image
        img = self.transforms(img)
        # print(triplet_imgs[0].shape=(3, 224, 224))

        label = self.class_ids[label]
        # batch = {
        # 'images': triplet_imgs, # list(torch.tensor(3, 224, 224)*3)
        # 'ids': triplet_ids, # nd.array([np.int64, np.int64, np.int64])
        # }
        return img, label

    def load_frames(self, path, boxes=None):
        # load images from paths
        # if boxes, crop corresponding box (x1,y1,x2,y2) from image
        return self.read_image(path, box=boxes)

    def read_image(self, image_name, box=None):
        r"""Return RGB image in PIL Image"""
        image = Image.open(os.path.join(self.img_path, image_name))
        if box is not None:
            image = image.crop(box)
        return image.convert("RGB")

    def sample(self, idx):
        """Sample path of triplet of images from the dataset"""
        # a_path, p_path, n_path = self.img_metadata['image_name'].iloc[idx].values
        path = self.img_metadata["image_name"][idx]
        # a_box, p_box, n_box = [self.img_metadata['box'].iloc[idx].values for _ in range(3)]
        boxes = self.img_metadata["box"][idx]
        label = self.img_metadata["label"][idx]
        return path, boxes, label

    def build_class_ids(self):
        """Build a dictionary of class ids"""
        class_ids = {}
        for i, class_name in enumerate(self.df["label"].unique()):
            class_ids[class_name] = i
        return class_ids

    def build_img_metadata(self):
        def read_metadata(df):
            """Return metadata
            metadata: Dictionary
                image_name
                box
                category_id
            """
            metadata = {
                "image_name": df["image_name"].values.tolist(),
                "box": df[["x_1", "y_1", "x_2", "y_2"]].values.tolist(),
                "category_id": df["category_id"].values.tolist(),
                "label": df["label"].values.tolist(),
            }
            return metadata

        def load_csv(split):
            with open(os.path.join(self.datapath, split + "_new_label.csv"), "r") as f:
                df = pd.read_csv(f)
                return df

        self.df = load_csv(self.split)

        img_metadata = {}
        if self.split in ["train", "val"]:
            img_metadata.update(read_metadata(self.df))
        else:
            raise Exception("Undefined split %s: " % self.split)

        print(
            "Total (%s) images are: %d" % (self.split, len(img_metadata["image_name"]))
        )
        return img_metadata


class DeepFashionOnlineTripletBalanceDataset(Dataset):
    def __init__(self, datapath, transforms, split):
        self.split = "val" if split in ["val", "test"] else "train"
        self.datapath = datapath
        self.img_path = os.path.join(self.datapath, self.split, "image")
        self.transforms = transforms
        self.batch_size = 128
        self.negative_percentage = 0.5
        self.img_metadata = self.build_img_metadata()
        self.class_ids = self.build_class_ids()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        # TODO: check this if data per batch is not loaded correctly
        paths, boxes, labels = self.sample(idx)
        # print("after load frames")
        # len(triplet_imgs)= 3 ; type(triplet_imgs) = list
        # type(triplet_imgs[0])=PIL Image
        # print(triplet_imgs[0].shape=(3, 224, 224))
        imgs = self.load_frames(paths=paths, boxes=boxes)
        imgs = [self.transforms(img) for img in imgs]

        labels = [self.class_ids[label] for label in labels]

        # batch = {
        # 'images': triplet_imgs, # list(torch.tensor(3, 224, 224)*3)
        # 'ids': triplet_ids, # nd.array([np.int64, np.int64, np.int64])
        # }
        imgs = torch.stack(imgs)
        labels = torch.Tensor(labels)
        return imgs, labels

    def load_frames(self, paths, boxes=None):
        # load images from paths
        # if boxes, crop corresponding box (x1,y1,x2,y2) from image
        frames = []
        for path, box in zip(paths, boxes):
            frames.append(self.read_image(path, box=box))
        return frames

    def read_image(self, image_name, box=None):
        r"""Return RGB image in PIL Image"""
        image = Image.open(os.path.join(self.img_path, image_name))
        if box is not None:
            image = image.crop(box)
        return image.convert("RGB")

    def sample(self, idx):
        """Sample path of triplet of images from the dataset"""
        # a_path, p_path, n_path = self.img_metadata['image_name'].iloc[idx].values
        pair_df = self.img_metadata[idx]

        missing_value = self.batch_size - len(pair_df)
        exclude_ids = pair_df.index.values

        if missing_value <= 0:
            exclude_ids = random.choices(
                exclude_ids, k=(self.batch_size * self.negative_percentage)
            )
            pair_df = pair_df[exclude_ids]

        random_ids = [
            random_exclusion(0, len(self.df) - 1, exclude_ids)
            for _ in range(missing_value)
        ]

        missing_df = self.df.iloc[random_ids, :]
        sample_df = pd.concat([pair_df, missing_df])

        paths = sample_df["image_name"].values.tolist()
        boxes = sample_df[["x_1", "y_1", "x_2", "y_2"]].values.tolist()
        labels = sample_df["label"].values.tolist()

        return paths, boxes, labels

    def build_class_ids(self):
        """Build a dictionary of class ids"""
        class_ids = {}
        for i, class_name in enumerate(self.df["label"].unique()):
            class_ids[class_name] = i
        return class_ids

    def build_img_metadata(self):
        def read_metadata(df):
            """Return metadata
            metadata: Dictionary
                image_name
                box
                category_id
            """
            metadata = {}
            groupby_df = df.groupby(by="pair_id")
            for i, (group, frame) in enumerate(groupby_df):
                metadata[i] = frame

            return metadata

        def load_csv(split):
            with open(os.path.join(self.datapath, split + "_new_label.csv"), "r") as f:
                df = pd.read_csv(f)
                return df

        self.df = load_csv(self.split)

        img_metadata = {}
        if self.split in ["train", "val"]:
            img_metadata.update(read_metadata(self.df))
        else:
            raise Exception("Undefined split %s: " % self.split)

        print("Total (%s) images are: %d" % (self.split, len(img_metadata)))
        return img_metadata
