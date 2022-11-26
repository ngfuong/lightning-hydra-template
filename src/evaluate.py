import os
import sys


# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
######################################################################
import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
#--------------------------------------------------------------------#

# Argparse 
parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument('--save_path', 
                    default='data\\model_inference\\shopping100k\\resnet50_1000\\ckpt5',
                    help='path to save the evaluation result file (.npy)')
# Path to the data embeddings
parser.add_argument('--emb_path', 
                    default='data\\model_inference\\shopping100k\\resnet50_1000\\ckpt5\\data_embeddings_ckpt5.npy',
                    help='Path to the embedding dataset')
# Top K nearest embedding (+1 for the query image)
parser.add_argument('--top', 
                    default=5,
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
    nn_model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn_model.fit(X)
    return nn_model


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
    
    if not os.path.exists(args.emb_path):
        raise ValueError(f"Embed dataset at path `{args.embed_path}` not found! Check your args.emb_path")
    emb_data = np.load(args.emb_path)
    print("Embedding data loaded. Embed data shape: ", emb_data.shape)

    dir_path = os.path.dirname(args.save_path)
    if not os.path.exists(dir_path):
        # Make containing directories
        os.makedirs(dir_path)
        
    # Create model for evaluation
    nn_model = kNN_model(emb_data, args.top)

    # Read attributes
    attr_path = open('data\Shopping100k\shorter_sim_attr.txt', 'r')
    lines = attr_path.readlines()
    attr_path.close()
    print(f"Finished reading file {attr_path}.")

    evaluate = []
    for line in lines:
        similar_list = line.split()
        similar_list = [int(i) for i in similar_list]
        # Generate one-hot encoding
        dists, indexes = nn_model.kneighbors(emb_data[similar_list[0],:].reshape(1,- 1), args.top)
        arr = np.isin(np.asarray(indexes[0]), np.asarray(similar_list)).tolist()
        evaluate.append(arr)

    np.save(args.save_path + '\evaluate{}.npy'.format(args.top-1), np.asarray(evaluate))
    print((f"Results saved to {args.save_path}."))
    
if __name__=="__main__":
    main()
