"""
In this script, we define the postprocessing steps to get the final predictions.
"""


########## Imports ##########
import numpy as np
import torch
from config import N_u, N_v


########## Functions ##########
def create_submission_matrix(predicted_ratings, submission_users, submission_items) -> torch.Tensor:
    """
    Create the submission matrix from the predicted ratings.
    """
    submission_matrix = np.zeros((N_u, N_v))
    submission_matrix[submission_users, submission_items] = predicted_ratings
    
    return submission_matrix

def reverse_standardize(submission_matrix, means, stds) -> torch.Tensor:
    reversed_ratings = submission_matrix * stds + means
    return reversed_ratings

########## Main ##########

def postprocess(model, submission_users, submission_items, means, stds):
    """
    Postprocess the model predictions.
    """
    model.eval()
    model.load_state_dict(torch.load("../data/models/best_val_model.pth")["model_state_dict"])
    raw_predicted_ratings = model.get_ratings(submission_users, submission_items).detach().cpu().numpy()
    raw_submission_matrix = create_submission_matrix(raw_predicted_ratings, submission_users, submission_items)
    submission_matrix = reverse_standardize(raw_submission_matrix, means, stds)
    prediction_ratings = submission_matrix[submission_users, submission_items]
    prediction_ratings = np.clip(prediction_ratings, 1, 5)