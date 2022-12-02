import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def visualize(indexes, dataframe, labels, dir, cols=5, save=False):
    """
    Use to plot images
    Args:
        indexes: list of indexes to access dataframe
        dataframe: dataframe contains image path,...
        labels: label of image you want to plot
        dir: directory to the image
        cols: number of columns you want
    Return:
        None
    """
    rows = len(indexes) // cols + 1
    for i in range(len(indexes)):
        image_name = dir + "/" + dataframe.loc[indexes[i], "image_name"]
        im = cv2.imread(image_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.imshow(im)
        plt.title(labels[indexes[i]])
        # plt.tight_layout()

    if save:
        plt.savefig("query.jpg")


def forward(x, model, device):
    x = x.type("torch.FloatTensor").to(device)
    return model(x)


def pil_loader(path, bbox=None):
    """
    Function to read image and crop if have bounding box
    Args:
        path: Path to the image
        bbox: list bounding box consist of [xmin, ymin, xmax, ymax]

    Return:
        PIL image
    """
    IMG = Image.open(path)
    if bbox:
        IMG = IMG.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    return IMG.convert("RGB")


def create_label_shopping100k(df):
    df["category_name"] = df["image_name"].apply(
        lambda x: " ".join(map(str, x.split("/")[1].split("_")[1:]))
    )
    labels = df["category_name"].values.tolist()
    return labels


def get_embed_model(num_classes, name="resnet"):
    """
    Function use to create embedding model
    Args:
        num_classes: output dim
        name: 'resnet' or 'vgg16bn' type of model
    Return:
        Embedding model
    """
    model = None
    if name == "resnet":
        model = torchvision.models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    return model


def kNN_model(X, k):
    """
    Function to train an NearestNeighbors model, use to improve the speed of retrieving image from embedding database
    Args:
        X: data to train has shape MxN
        k: number of max nearest neighbors want to search

    Return:
        Nearest Neigbors model
    """
    nn_model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn_model.fit(X)
    return nn_model


def get_transform_embed(im_size, train=False):
    """
    Function use for embedding to return method to Resize and Normalize image before feeding it to model
    Args:
        im_size: image size to resize image (Ex: 224)

    Return:
        transform function which contains Resize and Normalization
    """
    if train:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(im_size),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(im_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )


class Predictor:
    def __init__(self) -> None:
        df = pd.read_csv("data/shopping100k_similar.csv")
        # Create the category column
        self.labels = create_label_shopping100k(df)

        self.transforms = get_transform_embed((224, 224))

        self.emb_data = np.load("data_embeddings.npy")
        print(self.emb_data.shape)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # KNN model
        self.nn_model = kNN_model(self.emb_data, 30)

        # Create the embedding model and load checkpoint
        # Turn model to evaluation mode
        self.emb_model = get_embed_model(128)
        self.emb_model = self.emb_model.to(self.device)
        state_dict = torch.load("checkpoints/")["state_dict"]

        new_state_dict = state_dict.copy()

        for old_key, value in state_dict.items():
            new_state_dict[old_key.replace("net.", "")] = value
        del new_state_dict[old_key]
        print(new_state_dict.keys())
        self.emb_model.load_state_dict(new_state_dict)
        self.emb_model.eval()

    def predict(self, img_path):
        with torch.no_grad():
            image = pil_loader(img_path)

            # Embedding Resize and convert to tensor
            im = self.transforms(image)
            im = torch.unsqueeze(im, 0)
            # Embedding
            emb = forward(im, self.emb_model, self.device).cpu().numpy()
            dist, idx = self.nn_model.kneighbors(emb, 10)
        # Visualize images
        visualize(idx[0], self.df, self.labels, "data/Images", cols=5, save=True)
