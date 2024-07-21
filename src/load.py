"""
In this script we load the data.
"""

########## Imports ##########
import pandas as pd
from config import TRAIN_PATH, SUBMISSION_PATH

########## Functions ##########

def load_data():
    """
    Load the training data and the submission data.
    """
    train = pd.read_csv(TRAIN_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)

    return train, submission