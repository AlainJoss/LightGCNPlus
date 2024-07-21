"""
In this script, we run all necessary code to submit predictions to Kaggle.
This involves the following steps:
    - Load and preprocess the data
    - Train the model
    - Make predictions
    - Save the predictions to submission.csv
"""

########## Imports ##########
from load import load_data
from preprocess import preprocess
from train import train_model
from config import DEVICE, BASE_HYPERPARAMS
from models import ConcatNonLinear
import torch
from torch import nn

########## Functions ##########

def main():
    # Load
    train_df, submission_df = load_data()

    # Preprocess
    A_tilde, \
    standardized_train_ratings, \
    train_users, \
    train_items, \
    means, \
    stds, \
    val_users, \
    val_items, \
    standardized_val_ratings, \
    submission_users, \
    submission_movies = preprocess(train_df, submission_df)

    # Train
    L, K, INIT_EMBS_STD, LR, WEIGHT_DECAY, EPOCHS, STOP_THRESHOLD = BASE_HYPERPARAMS.values()
    model = ConcatNonLinear(A_tilde, K, L, INIT_EMBS_STD).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = train_model(model, optimizer, loss_fn, train_users, train_items, standardized_train_ratings, val_users, val_items, standardized_val_ratings, EPOCHS, STOP_THRESHOLD)

    # Post-process
    model.eval()
    model.load_state_dict(torch.load("../data/models/best_val_model.pth")["model_state_dict"])

