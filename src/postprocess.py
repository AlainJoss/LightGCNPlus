"""
The purpose of this script is to postprocess the model predictions.

This involves the following steps:
    1. Load the means and stds used for standardization.
    2. Load the best model from the training process.
    3. Get the raw predictions from the model for the submission users and items.
    4. Clip the predictions to the range [1, 5].
    5. Generate the submission file.

The following functions are defined:
    - load_means_stds: Load the means and stds used for standardization.
    - report_submission_results: Report the min, max, and mean of the final ratings.
    - report_clip_data: Report the min, max, and mean of the final ratings rounded.
    - create_submission_matrix: Create the submission matrix from the predicted ratings.
    - to_submission_format: Convert the submission data to the required format.
    - postprocess_report_submit: Postprocess the model predictions.
"""

########## Imports ##########
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns

from config import N_u, N_v
from load import load_submission_users_items
from models import load_best_val_model

########## Functions ##########

def load_means_stds(ID: str) -> tuple:
    path = f"../data/model_state/means_stds_{ID}.npz"
    with open(path, 'rb') as f:
        data = np.load(f)
        means = data['means']
        stds = data['stds']
    return means, stds

def report_submission_results(final_ratings: np.ndarray, rating_type: str) -> None:
    # Check min and max of final_ratings
    print("min:", final_ratings.min())
    print("max:", final_ratings.max())
    print("mean:", final_ratings.mean())

    # Check distribution of final_ratings
    plt.title(f"Distribution of {rating_type} final ratings")
    plt.hist(final_ratings.flatten(), bins=100)
    plt.show()

def report_clip_data(prediction_ratings):
    # check min and max of final_ratings_rounded
    print("min:", prediction_ratings.min().item())
    print("max:", prediction_ratings.max().item())
    print("mean:", prediction_ratings.mean().item())

    # Count the number of values under 1 and over 5
    count_under_1 = (prediction_ratings < 1).sum().item()
    count_over_5 = (prediction_ratings > 5).sum().item()
    print("count_over_5:", count_over_5)
    print("count_under_1:", count_under_1)
    print("")

def create_submission_matrix(predicted_ratings, submission_users, submission_items) -> torch.Tensor:
    """
    Create the submission matrix from the predicted ratings.
    """
    submission_matrix = np.zeros((N_u, N_v))
    submission_matrix[submission_users, submission_items] = predicted_ratings
    
    return submission_matrix

def to_submission_format(users, movies, predictions):
    return pd.DataFrame(data={'Id': ['r{}_c{}'.format(user + 1, movie + 1) for user, movie in zip(users, movies)],
                              'Prediction': predictions})

########## Main ##########


def report_training_results(train_rmse, val_rmse):
    # Replace values above 1 with 1 in the rmse lists
    train_rmse_plot = [min(1, x) for x in train_rmse]
    val_rmse_plot = [min(1, x) for x in val_rmse]

    # Set style
    sns.set(style="whitegrid")
    
    # Create plot
    plt.figure(dpi=300)  # Set maximum resolution
    plt.title("Training Results")
    plt.plot(train_rmse_plot, label='Training RMSE', color='darkred')
    plt.plot(val_rmse_plot, label='Validation RMSE', color='midnightblue')

    # Annotate min val loss
    min_val_rmse = min(val_rmse_plot)
    min_val_index = val_rmse_plot.index(min_val_rmse)
    plt.annotate(f"{round(min_val_rmse, 4)}", 
                 (min_val_index, min_val_rmse), 
                 textcoords="offset points", 
                 xytext=(0,-10), 
                 ha='center',
                 color='midnightblue')
    plt.axhline(y=min_val_rmse, color='midnightblue', linestyle='--', alpha=0.7)

    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


def postprocess_report_submit(model_class, ID):
    """
    Postprocess the model predictions.
    """
    sub_users, sub_items = load_submission_users_items()
    model = load_best_val_model(model_class, ID)

    # Get predictions from model for submission users and items
    raw_pred_ratings = model.get_ratings(sub_users, sub_items).detach().cpu().numpy()
    raw_submission_matrix = create_submission_matrix(raw_pred_ratings, sub_users, sub_items)
    pred_ratings = raw_submission_matrix[sub_users, sub_items]
    pred_ratings = np.clip(pred_ratings, 1, 5)

    # Report results
    report_submission_results(raw_pred_ratings, "raw")
    report_clip_data(pred_ratings)
    report_submission_results(pred_ratings, "clipped")

    # Generate submission file
    submission = to_submission_format(sub_users, sub_items, pred_ratings)
    submission.to_csv('../data/submission_data/submission.csv', index=False)