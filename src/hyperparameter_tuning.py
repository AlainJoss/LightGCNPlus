"""
In this script, we define the hyperparameter tuning for the model.
"""

########## Imports ##########
import json
import numpy as np
import torch
from torch import nn
from itertools import product
from train import train_model
from models import ConcatNonLinear
from config import DEVICE
from load import load_data
from preprocess import preprocess

########## Functions ##########
def report_combo(combo) -> None:
    """
    Print the hyperparameters combination.
    """
    print(f"HP: L={combo[0]}, K={combo[1]}, INIT_EMBS_STD={combo[2]}, LR={combo[3]}, WEIGHT_DECAY={combo[4]}")

def report_number_of_combos(grid) -> None:
    num_combinations = np.prod([len(v) for v in grid.values()])
    print(f"Number of combinations: {num_combinations}")


########## Hyperparams ##########

grid = {
    "L": [1, 2],
    "K": [30, 32, 34],
    "INIT_EMBS_STD": [0.01, 0.025, 0.05, 0.075, 0.1],
    "LR": [0.01, 0.025, 0.05, 0.075, 0.1],
    "WEIGHT_DECAY": [1e-04, 1e-05, 1e-06],
    "EPOCHS": [800],
    "STOP_THRESHOLD": [1e-06]
}


########## Main ##########
def hyperparameter_tuning(splits=3):

    results = []
    report_number_of_combos(grid)
    for combo in product(*grid.values()):
        # Unpack combo
        L, K, INIT_EMBS_STD, LR, WEIGHT_DECAY, EPOCHS, STOP_THRESHOLD = combo
        report_combo(combo)

        avg_best_val_loss = 0

        for _ in range(splits):
            # Load data
            train_df, submission_df = load_data()
            # Preprocess data
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
            submission_movies = preprocess((train_df, submission_df))

            # Create new model
            model = ConcatNonLinear(A_tilde, K, L, INIT_EMBS_STD).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            loss_fn = nn.MSELoss()

            # Train model
            train_losses, val_losses = train_model(model, optimizer, loss_fn, train_users, train_items, standardized_train_ratings, val_users, val_items, standardized_val_ratings, EPOCHS, STOP_THRESHOLD, save_best_model=False, hyper_verbose=False)

            # Sum to get cumulative best val loss
            avg_best_val_loss += min(val_losses)

        # Average losses over splits
        avg_best_val_loss /= splits
        print(f"--- Avg best val loss over splits: {avg_best_val_loss} ---")
            
        # Save results of run
        combo_results = {
            "combo": combo,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "avg_best_val_loss": avg_best_val_loss
        }
        results.append(combo_results)

        # Save intermediate results 
        with open('../data/logs/results.json', 'w') as f:
            json.dump(results, f)

    # Report best hyperparameters
    best_val_loss = min([result["avg_best_val_loss"] for result in results])
    best_combo = [result["combo"] for result in results if result["avg_best_val_loss"] == best_val_loss][0]
    print(f"Best avg val loss over runs: {best_val_loss}")
    print(f"Best avg combo over runs: {best_combo}")

if __name__ == "__main__":
    hyperparameter_tuning()