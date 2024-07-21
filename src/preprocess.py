"""
The purpose of this script is to preprocess the data for training the model.
This involves the following steps:
    - Extracting the users, movies, and ratings from the dataframes.
    - Standardizing the ratings matrix excluding unobserved values (zeros).
    - Splitting the data into training and validation sets.
    - Converting the data to torch tensors and moving them to the device.
    - Creating the normalized bipartite graph adjacency matrix for message passing.
    - Returning the normalized adjacency matrix and the train/val data for training.
"""

########## Imports ##########
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import DEVICE, N_u, N_v, VAL_SIZE

########## Functions ##########
def extract_users_items_ratings(df):
    """
    Extract users, movies, and predictions from the dataframe
    """
    users, movies = [np.squeeze(arr) for arr in np.split(df.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    ratings = df.Prediction.values

    return users, movies, ratings


def standardize_excluding_zeros(rating_matrix: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Standardize ratings excluding unobserved values (zeros).
    """
    # Compute means and stds of non-zero elements
    sums = np.sum(rating_matrix * mask, axis=0, keepdims=True)
    counts = np.sum(mask, axis=0, keepdims=True)
    means = sums / counts
    variances = np.sum(((rating_matrix - means) * mask) ** 2, axis=0, keepdims=True) / counts
    stds = np.sqrt(variances)
    standardized_ratings = (rating_matrix - means) / stds

    # Preserve original zeros
    standardized_ratings[~mask] = 0  

    # For simplicity make matrix of means and stds with same shape as ratings
    means = np.repeat(means, rating_matrix.shape[0], axis=0)
    stds = np.repeat(stds, rating_matrix.shape[0], axis=0)

    return standardized_ratings, means, stds

def create_bipartite_graph(users: torch.Tensor, items: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
    """
    Create a bipartite graph from the users, items and ratings.
        graph: [ [0, R], 
                 [R^T, 0] ] 
        shape: [ [N_u x N_u], [N_u x N_v], 
                 [N_v x N_u], [N_v x N_v] ]
    """
    # Create the for submatrices of the bipartite graph
    upper_left = torch.zeros((N_u, N_u), device=DEVICE)
    
    upper_right = torch.zeros((N_u, N_v), device=DEVICE)
    upper_right[users, items] = ratings

    lower_left = torch.zeros((N_v, N_u), device=DEVICE)
    lower_left[items, users] = ratings

    lower_right = torch.zeros((N_v, N_v), device=DEVICE)

    # Concatenate the submatrices to get the bipartite graph
    upper_block = torch.cat([upper_left, upper_right], dim=1)
    lower_block = torch.cat([lower_left, lower_right], dim=1)
    bip_adj_matrix = torch.cat([upper_block, lower_block], dim=0)
    return bip_adj_matrix

def create_degree_matrix(bip_adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Create the degree matrix of the bipartite graph.
    """
    # Binarize rating matrix for computing the degrees
    binarized_adjacency_matrix = (bip_adj_matrix != 0).int()
    # Sum the binarized adjacency matrix along axis 1 to get the degree of each node
    degrees = torch.sum(binarized_adjacency_matrix, dim=1)
    # Create a diagonal matrix with the degrees
    degree_matrix = torch.diag(degrees)
    return degree_matrix

def create_inverse_sqrt_degree_matrix(bip_degree_matrix: torch.Tensor) -> torch.Tensor:
    """
    Create the inverse square root degree matrix of the bipartite graph.
    """
    # Get the degrees from the diagonal of the degree matrix
    degrees = bip_degree_matrix.diag()
    # Compute the inverse square root of the degrees
    inverse_sqrt_degrees = 1.0 / torch.sqrt(degrees)
    # Create a diagonal matrix with the inverse square root degrees
    inverse_sqrt_degree_matrix = torch.diag(inverse_sqrt_degrees)
    return inverse_sqrt_degree_matrix

########## Main ##########

def preprocess(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple:
    """
    Get the normalized bipartite adjacency matrix for training.
    """
    # Load data
    train_df, submission_df = data

    # Extract adjacency lists: observed values edge index (src, tgt) and ratings (values)
    submission_users, submission_items, _ = extract_users_items_ratings(submission_df)
    train_users, train_items, train_ratings = extract_users_items_ratings(train_df)

    # Create rating matrix from the triplets
    train_ratings_matrix = np.zeros((N_u, N_v))
    train_ratings_matrix[train_users, train_items] = train_ratings

    # Define mask for selecting observed values
    mask = train_ratings_matrix != 0

    # Standardize the ratings matrix across columns (items) and extract ratings list for observed values
    standardized_rating_matrix, means, stds = standardize_excluding_zeros(train_ratings_matrix, mask)
    standardized_ratings = standardized_rating_matrix[train_users, train_items]

    # Split the data into trai and val sets
    train_users, val_users, train_items, val_items, standardized_train_ratings, standardized_val_ratings = \
        train_test_split(train_users, train_items, standardized_ratings, test_size=VAL_SIZE)
    
    # Get the original val ratings for evaluation
    original_val_ratings = train_ratings_matrix[val_users, val_items]
    
    # Convert to torch tensors for training and move to device
    
    # convert standardized_rating_matrix, means and stds to torch tensors

    standardized_train_ratings = torch.tensor(standardized_train_ratings, dtype=torch.float).to(DEVICE)
    train_users = torch.tensor(train_users, dtype=torch.long).to(DEVICE)
    train_items = torch.tensor(train_items, dtype=torch.long).to(DEVICE)

    standardized_val_ratings = torch.tensor(standardized_val_ratings, dtype=torch.float).to(DEVICE)
    original_val_ratings = torch.tensor(original_val_ratings, dtype=torch.float).to(DEVICE)
    val_users = torch.tensor(val_users, dtype=torch.long).to(DEVICE)
    val_items = torch.tensor(val_items, dtype=torch.long).to(DEVICE)

    # Create the bipartite graph adjacency matrix
    bip_adj_matrix = create_bipartite_graph(train_users, train_items, standardized_train_ratings)
    D = create_degree_matrix(bip_adj_matrix)
    D_norm = create_inverse_sqrt_degree_matrix(D)

    # Compute the normalized adjacency matrix and move to device
    A_tilde = D_norm @ bip_adj_matrix @ D_norm
    A_tilde = A_tilde.to(DEVICE)

    return A_tilde, standardized_train_ratings, train_users, train_items, means, stds, val_users, val_items, original_val_ratings, standardized_val_ratings, submission_users, submission_items