"""
In this script we load the data.
"""

########## Imports ##########
import pandas as pd
import pickle
from config import TRAIN_PATH, SUBMISSION_PATH

########## Functions ##########

def load_train_data():
    """
    Load the training data and the submission data.
    """
    train_df = pd.read_csv(TRAIN_PATH)

    return train_df

def load_submission_users_items():
    """
    Load the submission users and items.
    """
    with open(SUBMISSION_PATH, 'rb') as f:
        submission_users, submission_items = pickle.load(f)

    return submission_users, submission_items