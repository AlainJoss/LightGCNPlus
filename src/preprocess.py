"""
The purpose of this script is to preprocess the data for training the model.

This involves the following steps:
    1. Extract users, movies, and ratings from the dataframe.
    2. Split the data into training and validation sets.
    3. Create the rating matrix from the training triplets.
    4. Define a mask for selecting observed values in the training set.
    5. Standardize the training ratings matrix across columns (items).
    6. Standardize the training ratings matrix across rows (users).
    7. Convert the data to torch tensors for training and move to device.
    8. Create the bipartite graph adjacency matrix from the training data.
    9. Create the degree matrix of the bipartite graph.
    10. Create the inverse square root degree matrix of the bipartite graph.
    11. Compute the normalized adjacency matrix and move to device.
    12. Return the normalized bipartite adjacency matrix,
        together with the training and validation users, items, and ratings.

The following functions are defined:
    - extract_users_items_ratings: Extract users, movies, and predictions from the dataframe.
    - standardize: Standardize ratings excluding unobserved values (zeros).
    - create_bipartite_graph: Create a bipartite graph from the users, items and ratings.
"""

########## Imports ##########
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import DEVICE, N_u, N_v, VAL_SIZE

########## Functions ##########
def extract_users_items_ratings(df):
    """
    Extract users, movies, and predictions from the dataframe.
    """
    users, movies = [np.squeeze(arr) for arr in np.split(df.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    ratings = df.Prediction.values
    return users, movies, ratings

def standardize(rating_matrix: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """
    Standardize ratings excluding unobserved values (zeros).
    """
    # Compute means and stds of non-zero elements
    sums = np.sum(rating_matrix * mask, axis=axis, keepdims=True)
    counts = np.sum(mask, axis=axis, keepdims=True)
    means = sums / counts
    variances = np.sum(((rating_matrix - means) * mask) ** 2, axis=axis, keepdims=True) / counts
    stds = np.sqrt(variances)

    # Standardize the ratings matrix
    z_score_ratings = (rating_matrix - means) / stds

    # Overwrite unobserved values with zeros.
    # These values will not be used in training, but will still be used when computing A_tilde.
    # Thus, we still need them to have values, which do mean something different to zero in the standardized space.
    z_score_ratings[~mask] = 0

    return z_score_ratings

def create_bipartite_graph(users: torch.Tensor, items: torch.Tensor, col_z_score_ratings: torch.Tensor, row_z_score_ratings) -> torch.Tensor:
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
    upper_right[users, items] = col_z_score_ratings

    lower_left = torch.zeros((N_v, N_u), device=DEVICE)
    lower_left[items, users] = row_z_score_ratings

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

def preprocess(train_df: pd.DataFrame) -> tuple:
    """
    Get the normalized bipartite adjacency matrix for training.
    """
    # Extract adjacency lists: observed values edge index (src, tgt) and ratings (values)
    all_users, all_items, all_ratings = extract_users_items_ratings(train_df)

    # Split the data into training and validation sets
    train_users, val_users, train_items, val_items, train_ratings, val_ratings = \
        train_test_split(all_users, all_items, all_ratings, test_size=VAL_SIZE)

    # Create rating matrix from the training triplets
    train_ratings_matrix = np.zeros((N_u, N_v))
    train_ratings_matrix[train_users, train_items] = train_ratings

    # Define mask for selecting observed values in the training set
    train_mask = train_ratings_matrix != 0

    # Standardize the training ratings matrix across columns (items)
    col_z_score_rating_matrix = standardize(train_ratings_matrix, train_mask, axis=0)
    col_z_score_ratings = col_z_score_rating_matrix[train_users, train_items]

    # Standardize the training ratings matrix across rows (users)
    row_z_score_rating_matrix = standardize(train_ratings_matrix, train_mask, axis=1)
    row_z_score_ratings = row_z_score_rating_matrix[train_users, train_items]

    # Convert to torch tensors for training and move to device
    col_z_score_ratings = torch.tensor(col_z_score_ratings, dtype=torch.float).to(DEVICE)
    row_z_score_ratings = torch.tensor(row_z_score_ratings, dtype=torch.float).to(DEVICE)

    train_ratings = torch.tensor(train_ratings, dtype=torch.float).to(DEVICE)
    train_users = torch.tensor(train_users, dtype=torch.long).to(DEVICE)
    train_items = torch.tensor(train_items, dtype=torch.long).to(DEVICE)

    val_ratings = torch.tensor(val_ratings, dtype=torch.float).to(DEVICE)
    val_users = torch.tensor(val_users, dtype=torch.long).to(DEVICE)
    val_items = torch.tensor(val_items, dtype=torch.long).to(DEVICE)

    # Create the bipartite graph adjacency matrix from the training data
    bip_adj_matrix = create_bipartite_graph(train_users, train_items, col_z_score_ratings, row_z_score_ratings)
    D_raw = create_degree_matrix(bip_adj_matrix)
    D_norm = create_inverse_sqrt_degree_matrix(D_raw)

    # Compute the normalized adjacency matrix and move to device
    A_tilde = D_norm @ bip_adj_matrix @ D_norm
    A_tilde = A_tilde.to(DEVICE)

    return A_tilde, train_users, train_items, train_ratings, val_users, val_items, val_ratings