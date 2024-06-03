import os.path as osp
import errno
import os
import numpy as np
import torch
from tqdm import tqdm



def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def get_knn_dists(val_base_class_features, image_features_cur, K_nns):

    print('do not exist the knn distances, calculate them')
    knndists = []

    # cpu version
    # # get current image fearture
    # image_features_cur = np.array(torch.cat(image_fea_list))
    # for feature in tqdm(image_features_cur, desc="Calculating image distance"):
    #     distances = np.linalg.norm(val_image_features - feature, axis=1)
    #     top_k_distances = np.sort(distances)[:K]  # top distance
    #     top_k_indices = np.argsort(distances)[:K]  # index
    #     knndists.append(top_k_distances)

    # gpu version
    image_features_cur = torch.tensor(image_features_cur, dtype=torch.float32).to('cuda')
    val_base_class_features = torch.tensor(val_base_class_features, dtype=torch.float32).to('cuda')

    for feature in tqdm(image_features_cur, desc="Calculating distance"):
        distances = torch.norm(val_base_class_features - feature, dim=1)
        top_k_distances, top_k_indices = torch.topk(distances, k=K_nns, largest=False) 

        top_k_distances = top_k_distances.cpu().numpy()
        knndists.append(top_k_distances)

    knndists = np.array(knndists)

    return knndists


def get_val_image_knn_dists(image_features_cur, K_nns):
    print('Calculating the K nearest neighbors distances in val image.')

    # Convert the image features to a PyTorch tensor and transfer to GPU
    image_features_cur = torch.tensor(image_features_cur, dtype=torch.float32).to('cuda')
    knndists = []

    # Iterate over each feature and calculate its distance to other features
    for index, feature in enumerate(tqdm(image_features_cur, desc="Calculating val image nn distance")):
        # Calculate distances from the current feature to all other features
        distances = torch.norm(image_features_cur - feature, dim=1)

        # We sort the distances and pick the top K, excluding the distance to itself (which is 0)
        top_k_distances, top_k_indices = torch.topk(distances, k=K_nns+1, largest=False)

        # Exclude the zero distance (distance to itself)
        top_k_distances = top_k_distances[1:]  # Skip the first one as it is the distance to itself
        top_k_distances = top_k_distances.cpu().numpy()
        knndists.append(top_k_distances)

    knndists = np.array(knndists)
    return knndists
