"""
The purpose of this script is to define the configuration variables used in the project.

The following variables are defined:
    - DEVICE: The device to use for training the model.
    - N_u, N_v: The number of users and items in the dataset.
    - TRAIN_PATH: The path to the training data file.
    - SUBMISSION_PATH: The path to the submission data file.
    - VAL_SIZE: The size of the validation set as a fraction of the training data.
"""

import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_u, N_v = (10000, 1000)

TRAIN_PATH = "../data/raw_data/train.csv"
SUBMISSION_PATH = "../data/submission_data/submission_users_items.pkl"

VAL_SIZE = 0.01019582787  # 12000 / 1176952