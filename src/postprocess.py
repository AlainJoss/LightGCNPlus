"""
In this script, we define the postprocessing steps to get the final predictions.
"""

########## Imports ##########
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from config import N_u, N_v
from load import load_submission_users_items
from models import load_best_val_model

########## Functions ##########

def report_submission_results(final_ratings, rating_type):
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

def reverse_standardize(submission_matrix, means, stds) -> torch.Tensor:
    reversed_ratings = submission_matrix * stds + means
    return reversed_ratings

def to_submission_format(users, movies, predictions):
    return pd.DataFrame(data={'Id': ['r{}_c{}'.format(user + 1, movie + 1) for user, movie in zip(users, movies)],
                              'Prediction': predictions})

########## Main ##########

def report_training_results(train_rmse, val_rmse_std, val_rmse_orig):
    # Training stats    
    print("Min training loss:", round(min(train_rmse), 4))
    print("Min validation loss std:", round(min(val_rmse_std), 4))
    print("Min validation loss orig:", round(min(val_rmse_orig), 4))
    print("Min validation loss at epoch:", val_rmse_std.index(min(val_rmse_std)))

    # Replace values above 1 with 1 in the rmse lists
    train_rmse_plot = [min(1, x) for x in train_rmse]
    val_rmse_std_plot = [min(1, x) for x in val_rmse_std]
    val_rmse_orig_plot = [min(1, x) for x in val_rmse_orig]

    plt.title("Training Results")
    # Plot train and val rmse
    plt.plot(train_rmse_plot, label='train')
    plt.plot(val_rmse_std_plot, label='val std')
    plt.plot(val_rmse_orig_plot, label='val orig')
    plt.plot()

    # Annotate min val loss
    plt.annotate(round(min(val_rmse_orig_plot), 4), (val_rmse_orig_plot.index(min(val_rmse_orig_plot)), min(val_rmse_orig_plot)), textcoords="offset points", xytext=(0,-10), ha='center')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

def postprocess(model, means, stds):
    """
    Postprocess the model predictions.
    """
    # Read model that achieved best validation loss
    submission_users, submission_items = load_submission_users_items()
    # Load model inputs
    model = load_best_val_model()
    model.eval()
    # Get predictions for submission
    final_ratings = model.get_ratings(submission_users, submission_items).cpu().detach().numpy()
    
    report_submission_results(final_ratings, "raw")
    
    raw_predicted_ratings = model.get_ratings(submission_users, submission_items).detach().cpu().numpy()
    raw_submission_matrix = create_submission_matrix(raw_predicted_ratings, submission_users, submission_items)
    submission_matrix = reverse_standardize(raw_submission_matrix, means, stds)
    prediction_ratings = submission_matrix[submission_users, submission_items]
    
    report_clip_data(prediction_ratings)
    prediction_ratings = np.clip(prediction_ratings, 1, 5)

    report_submission_results(prediction_ratings, "clipped")

    submission = to_submission_format(submission_users, submission_items, prediction_ratings)
    submission.to_csv('../data/submission_data/submission.csv', index=False)