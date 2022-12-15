import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

import numpy as np
import pandas as pd
import torch

##################################################
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from PIL import Image
from torch.autograd import Variable

# Define variables
BATCH_SIZE = 256

parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument(
    "--df_path",
    default="data/shopping100k_similar.csv",
    help="Dataframe contains the deep fashion dataset",
)
# Directory to the image dir
parser.add_argument(
    "--img_dir", 
    default="data/shopping100k/Images", 
    help="Path to Shopping100k Images directory"
)
# Path to the embedding model state dict
parser.add_argument(
    "--emb",
    default="checkpoints/resnet101/top_k_acc=0.715.ckpt",
    help="Path to the embedding model state dict",
)
# Output path of enbedding
parser.add_argument(
    "--save_dir",
    default="checkpoints/embeddings",
    help="Path to save file embedding",
)


def pil_loader(element):
    IMG = Image.open(element)

    return IMG.convert("RGB")


class Shopping100k(torch.utils.data.Dataset):
    """Custom dataset to load Shopping100k dataset for testing"""

    def __init__(self, df, im_size, root_dir, transform=None, loader=pil_loader):
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

        self.df["image_name"] = self.df["image_name"].apply(
            lambda x: os.path.join(self.root_dir, x)
        )

    def _sample(self, idx):
        p = self.df.loc[idx, "image_name"]
        return p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self._sample(idx)

        temp = self.loader(path)
        temp = self.transform(temp)
        return temp, 0


def main():
    # parse the variables
    args = parser.parse_args()
    """
    1. Read csv
    2. Load dataset
    3. Feed into DataLoader
    """
    df = pd.read_csv(args.df_path)
    transforms = transform.Compose(
        [
            transform.Resize(size=(224, 224)),
            transform.ToTensor(),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_dataset = Shopping100k(
        df, im_size=(224, 224), root_dir=args.img_dir, transform=transforms
    )
    evalloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the embedding model and load checkpoint
    emb_model = torchvision.models.resnet50(pretrained=True)
    num_features = emb_model.fc.in_features
    emb_model.fc = nn.Linear(num_features, 128)

    emb_model = emb_model.to(device)
    state_dict = torch.load(args.emb)["state_dict"]

    new_state_dict = state_dict.copy()

    for old_key, value in state_dict.items():
        new_state_dict[old_key.replace("net.", "")] = value
        del new_state_dict[old_key]
    print(new_state_dict.keys())
    emb_model.load_state_dict(new_state_dict)
    emb_model.eval()

    """
    1. Move batch to cuda
    2. Run through model
    3. concat to embedding matrix
    4. Save embedding matrix to file
    """
    embedding = torch.randn(1, 2000).type("torch.FloatTensor").to(device)
    with torch.no_grad():
        for batch_idx, (eval_image, _) in enumerate(evalloader):
            eval_image = Variable(eval_image).cuda()
            emb = emb_model(eval_image)
            embedding = torch.cat((embedding, emb), 0)

    embedding = np.delete(embedding.cpu().numpy(), np.s_[:1], axis=0)
    np.save(args.save_dir + "/data_embeddings", embedding)


if __name__ == "__main__":
    main()
