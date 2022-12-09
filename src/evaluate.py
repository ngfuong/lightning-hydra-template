import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
######################################################################
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import torch
import torchvision
import torch.nn as nn
from src.utils.metrics import TopKAccuracy, MeanReciprocalRank
from src.datamodules.datasets import OnlineTripletDataset
from src.models.online_triplet_module import OnlineTripletModule
from src.models.components.convnet import ConvNet_VGG16bn
#--------------------------------------------------------------------#

# Argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--weights",
                    required=True,
                    type=str)
parser.add_argument('--backbone',
                    default="resnet50")
parser.add_argument('--base_lr',
                    default="1e-3")
parser.add_argument('--batch_size',
                    default=16,
                    type=int)
parser.add_argument('--max_epochs',
                    default=30,
                    type=int)
parser.add_argument('--data_path',
                    default="data/")
# Top K nearest embedding (+1 for the query image)
parser.add_argument('--top_k', 
                    default=10,
                    type=int,
                    help='Top K nearest embedding (+1 for the query image)')                    
                    
def kNN_model(X, k):
    """
    Function to train an NearestNeighbors model, use to improve the speed of retrieving image from embedding database
    Args:
        X: data to train has shape MxN
        k: number of max nearest neighbors want to search
    
    Return:
        Nearest Neigbors model
    """
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(X)
    return nn_model

def get_embed_model(num_classes, name):
    """
    Function use to create embedding model
    Args:
        num_classes: output dim
        name: 'resnet' or 'vgg16bn' type of model
    Return:
        Embedding model
    """
    model = None
    if name == 'resnet50':
        model =  torchvision.models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features,num_classes)
    elif name == 'resnet101':
        model =  torchvision.models.resnet101(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features,num_classes)
    elif name == 'vgg16bn':
        model =  ConvNet_VGG16bn(num_classes)
    return model


def main():
    # parse the variables
    args = parser.parse_args()

    """
    1. Open file similar attributes
    2. Load the dataset embedding
    3. Load the K NN model
    4. Find whether similar attribute image is in the query or not
    5. Save the evaluate
    """

    module_def = OnlineTripletModule

    module = module_def.load_from_checkpoint(
        checkpoint_path = args.weights,
        data_path = args.data_path,
        data_set = "deepfashion_val",
        batch_size = 1,
        backbone = args.backbone,
    )
    
    OnlineTripletDataset.initialize(
        img_size=224, datapath=args.data_path, imagenet_norm=True
    )

    query_dataloader = OnlineTripletDataset.build_dataloader(
        benchmark="deepfashion_val", 
        bsz = 1, 
        nworker = 8, 
        split = "val", 
        val_type = "query"
    )

    gallery_dataloader = OnlineTripletDataset.build_dataloader(
        benchmark="deepfashion_val",
        bsz = 1, 
        nworker = 8, 
        split = "val", 
        val_type = "gallery"
        )

    gallery_vectors = []
    gallery_classes = []
    class_ids = {}
    class_count = 0

    # emb_model = get_embed_model(128, name=args.backbone)
    # emb_model = emb_model.to("cuda")
    # state_dict = torch.load(args.weights)["state_dict"]
    
    # new_state_dict = state_dict.copy()

    # for old_key, value in state_dict.items():
    #   new_state_dict[old_key.replace("net.", "")] = value
    #   del new_state_dict[old_key]
    # print(new_state_dict.keys())
    # emb_model.load_state_dict(new_state_dict)
    emb_model = module.net.eval().to("cuda")

    print("Embedding dataset...")
    for batch_idx, batch in tqdm(enumerate(gallery_dataloader), total=len(gallery_dataloader)):
        imgs, pair_ids, styles = batch
        imgs = torch.stack(imgs, 0).to("cuda")

        feature_vector = emb_model(imgs)
        gallery_vectors.append(feature_vector.to("cpu"))
        # label
        for i in range(len(pair_ids)):
            label = f"{pair_ids[i]}_{styles[i]}"
            if label not in class_ids:
                class_ids[f"{pair_ids[i]}_{styles[i]}"] = class_count
                class_count += 1
            gallery_classes.append(class_ids[label])

    gallery_vectors = torch.cat(gallery_vectors, 0)
    
    knn = NearestNeighbors(n_neighbors=20)
    knn.fit(gallery_vectors)

    class_ids = class_ids
    gallery_classes = torch.Tensor(gallery_classes)

    top_k = args.top
    top_k_accuracy = TopKAccuracy()
    mean_reciprocal_rank = MeanReciprocalRank()
    print("Evaluation...")
    with torch.no_grad():
        for batch in tqdm(query_dataloader, total=len(query_dataloader)):
            imgs, pair_ids, styles = batch
            imgs = torch.stack(imgs, 0).to("cuda")
            embeddings = emb_model(imgs)

            # Convert label
            labels = []
            assert len(pair_ids) == len(styles)
            for i in range(len(pair_ids)):
                label = class_ids[f"{pair_ids[i]}_{styles[i]}"]
                labels.append(label)

            dists, indexes = knn.kneighbors(embeddings.to("cpu"), top_k)
            top_k_classes = gallery_classes[indexes]
            if len(top_k_classes.shape) == 1:
                top_k_classes = torch.unsqueeze(top_k_classes, dim=0)
            # calculate top k acc
            top_k_accuracy.update(labels, top_k_classes)
            #  mean reciprocal rank
            mean_reciprocal_rank.update(labels, top_k_classes)

    top_k_acc = top_k_accuracy.compute()
    mrr = mean_reciprocal_rank.compute()
    print(f"Top {top_k} accuracy: {top_k_acc}")
    print(f"MRR {top_k}: {mrr}")

if __name__=="__main__":
    main()
