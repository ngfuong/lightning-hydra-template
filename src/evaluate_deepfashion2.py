import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

import numpy as np
import pandas as pd

##################################################
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable
from tqdm import tqdm

from src.datamodules.onlinetriplets import DeepFashionOnlineValidationDataset

# Define variables
BATCH_SIZE = 600

parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument(
    "--df_path",
    default="data/list_bbox_consumer2shop.txt",
    help="Path to DeepFashion file.",
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


def main():
    # parse the variables
    args = parser.parse_args()
    """
    1. Read csv
    2. Load dataset
    3. Feed into DataLoader
    """
    df = pd.read_csv(args.df_path, skiprows=1, delimiter="\s+")
    transforms = transform.Compose(
        [
            transform.Resize(size=(224, 224)),
            transform.ToTensor(),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # eval_dataset = DeepFashionGallery(df, im_size=(224, 224), root_dir=args.img_dir, source_type=1)
    eval_dataset = DeepFashionOnlineValidationDataset(
        datapath="./data", transforms=transforms, split="val", val_type="gallery"
    )
    evalloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # query_dataset = DeepFashionGallery(df, im_size=(224, 224), root_dir=args.img_dir, source_type=2)
    query_dataset = DeepFashionOnlineValidationDataset(
        datapath="./data", transforms=transforms, split="val", val_type="query"
    )
    queryloader = torch.utils.data.DataLoader(
        query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the embedding model and load checkpoint
    emb_model = torchvision.models.resnet101(pretrained=True)
    num_features = emb_model.fc.in_features
    emb_model.fc = nn.Linear(num_features, 128)

    emb_model = emb_model.to(device)
    state_dict = torch.load(args.emb, map_location=device)["state_dict"]

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
    embedding = torch.randn(1, 128).type("torch.FloatTensor").to(device)
    label_list = []

    class_ids = {}
    class_count = 0

    with torch.no_grad():
        for batch_idx, (eval_image, pair_ids, styles) in tqdm(enumerate(evalloader), total=len(evalloader)):
            print(eval_image.shape)
            for i in range(len(pair_ids)):
                label = f"{pair_ids[i]}_{styles[i]}"
                print(label)
                if label not in class_ids:
                    class_ids[f"{pair_ids[i]}_{styles[i]}"] = class_count
                    class_count += 1
                label_list.append(class_ids[label])
    
            if torch.cuda.is_available():
                eval_image = Variable(eval_image).cuda()
            else:
                eval_image = Variable(eval_image).cpu()
            emb = emb_model(eval_image)
            embedding = torch.cat((embedding, emb), 0)

        embedding = np.delete(embedding.cpu().numpy(), np.s_[:1], axis=0)
        np.save("data_embedding.npy", embedding)
        # embedding = np.load('data_embedding.npy')
        nn_model = kNN_model(embedding, 30)
        label_list = torch.Tensor(label_list)
        print(label_list.shape)

        top_k_acc = [0 for i in range(31)]
        mrr = [0 for i in range(31)]
        total = 0
        for batch_idx, (query_image, pair_ids, styles) in tqdm(enumerate(queryloader), total=len(queryloader)):
            labels = []
            assert len(pair_ids) == len(styles)
            for i in range(len(pair_ids)):
                label = class_ids[f"{pair_ids[i]}_{styles[i]}"]
                labels.append(label)

            if torch.cuda.is_available():
                eval_image = Variable(eval_image).cuda()
            else:
                eval_image = Variable(eval_image).cpu()
            emb = emb_model(query_image)
            dist, idx = nn_model.kneighbors(emb.cpu(), 30)

            for i in range(len(idx)):
                current_label = labels[i]
                for k in range(1, 31):
                    current_idx = idx[i, :k]
                    gallery_class = label_list[current_idx]

                    # print(current_label, gallery_class)
                    isin = (gallery_class == current_label).nonzero()
                    # print(isin)
                    if len(isin) > 0:
                        mrr[k] += 1 / (isin[0].item() + 1)
                        top_k_acc[k] += 1
                total += 1
                # print(mrr, top_k_acc, total)

        for k in range(1, 31):
            print(f"TOP {k} ACC: ", top_k_acc[k] / total)
            print(f"MRR {k}: ", mrr[k] / total)

        df = pd.DataFrame()
        df["k"] = [i for i in range(31)]
        df["top_k_acc"] = top_k_acc
        df["mrr"] = mrr
        df.to_csv("evaluation.csv", index=False)

    # np.save(args.save_dir+'/data_embeddings',embedding)


if __name__ == "__main__":
    main()
