import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

import numpy as np
import pandas as pd

##################################################
import torch
from torch.autograd import Variable

from utils.data_utils import Shopping100k
from utils.model_utils import get_embed_model
from utils.search_utils import *

# Define variables
BATCH_SIZE = 256

parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument(
    "--df_path",
    default="data\Shopping100k\Attributes\shopping100k.csv",
    help="Dataframe contains the deep fashion dataset",
)
# Directory to the image dir
parser.add_argument(
    "--img_dir", default="data\Shopping100k\Images", help="Root dir to the image dir"
)
# Path to the embedding model state dict
parser.add_argument(
    "--emb",
    default="fashion-visual-search\src\pytorch\models\embedding_model\multinet_VGG16bn\multi_net_ckpt11.pt",
    help="Path to the embedding model state dict",
)
# Output path of enbedding
parser.add_argument(
    "--save_dir",
    default="data\model_inference\shopping100k\Multinet\ckpt11",
    help="Path to save file embedding",
)


def main():
    # parse the variables
    args = parser.parse_args()
    """
    1. Read csv
    2. Load dataset
    3. Feed into DataLoader
    """
    df = pd.read_csv(args.df_path)
    eval_dataset = Shopping100k(df, im_size=(224, 224), root_dir=args.img_dir)
    evalloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the embedding model and load checkpoint
    emb_model = get_embed_model(2000)
    emb_model = emb_model.to(device)
    emb_model.load_state_dict(torch.load(args.emb))
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


# KNN
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
######################################################################
import argparse

import numpy as np
import pandas as pd

from utils.search_utils import kNN_model

# --------------------------------------------------------------------#

# Argparse
parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument(
    "--save_path",
    default="data\\model_inference\\shopping100k\\resnet50_1000\\ckpt5",
    help="path to the save file",
)
# Path to the data embeddings
parser.add_argument(
    "--emb_path",
    default="data\\model_inference\\shopping100k\\resnet50_1000\\ckpt5\\data_embeddings_ckpt5.npy",
    help="Path to the embedding dataset",
)
# Top K nearest embedding (+1 for the query image)
parser.add_argument(
    "--top",
    default=71,
    type=int,
    help="Top K nearest embedding (+1 for the query image)",
)


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
    similar_attr = open("data\Shopping100k\shorter_sim_attr.txt", "r")
    lines = similar_attr.readlines()
    print(len(lines))

    emb_data = np.load(args.emb_path)
    print(emb_data.shape)

    nn_model = kNN_model(emb_data, args.top)

    evaluate = []
    for line in lines:
        similar_list = line.split()
        similar_list = [int(i) for i in similar_list]
        dists, indexes = nn_model.kneighbors(
            emb_data[similar_list[0], :].reshape(1, -1), args.top
        )
        arr = np.isin(np.asarray(indexes[0]), np.asarray(similar_list)).tolist()
        evaluate.append(arr)

    np.save(
        args.save_path + "\evaluate{}.npy".format(args.top - 1), np.asarray(evaluate)
    )
    similar_attr.close()


if __name__ == "__main__":
    main()


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
