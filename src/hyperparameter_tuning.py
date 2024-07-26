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
from models import ConcatNonLinear_41out, ConcatNonLinear_42out, ConcatNonLinear_421out
from config import DEVICE, SPLITS
from load import load_data
from preprocess import preprocess

########## Functions ##########
def report_run(model_name, combo, counter, total_runs) -> None:
    """
    Print the hyperparameters combination.
    """
    print(f"Run {counter}/{total_runs}")
    print(f"Model: {model_name}")
    print(f"HP: L={combo[0]}, K={combo[1]}, INIT_EMBS_STD={combo[2]}, LR={combo[3]}, WEIGHT_DECAY={combo[4]}, DROPOUT={combo[5]}")

def num_combos(grid) -> None:
    num_combinations = np.prod([len(v) for v in grid.values()]) * len(models) * SPLITS
    print(f"Number of combinations: {num_combinations}")
    return num_combinations

########## Hyperparams ##########

grid = {
    "L": [1],
    "K": [30],
    "INIT_EMBS_STD": [0.025, 0.05, 0.075, 0.1],
    "LR": [0.05, 0.075, 0.1],
    "WEIGHT_DECAY": [1e-05],
    "DROPOUT": [0.3, 0.4, 0.5],
    "EPOCHS": [200],
    "STOP_THRESHOLD": [1e-06]
}

models = {
    "ConcatNonLinear_41out": ConcatNonLinear_41out,
    "ConcatNonLinear_42out": ConcatNonLinear_42out,
    "ConcatNonLinear_421out": ConcatNonLinear_421out
}

########## Main ##########

def tune_hyperparameters(models: list, grid: dict) -> None:
    counter = 0
    n_combos = num_combos(grid)
    for model_name, mod in models.items():
        for combo in product(*grid.values()):
            counter += 1
            report_run(model_name, combo, counter, n_combos)
            


def hyperparameter_tuning(splits):

    count_runs = 0
    results = []
    n_combos = num_combos(grid)
    for model_name, mod in models.items():
        print(f"Model: {model_name}")

        for combo in product(*grid.values()):
            # Unpack combo
            L, K, INIT_EMBS_STD, LR, WEIGHT_DECAY, DROPOUT, EPOCHS, STOP_THRESHOLD = combo
            report_combo(combo)

            avg_best_val_loss = 0

            for _ in range(splits):
                count_runs += 1
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
                original_val_ratings, \
                standardized_val_ratings, \
                _, \
                _ = preprocess((train_df, submission_df))

                # Create new model
                model = mod(A_tilde, K, L, INIT_EMBS_STD, DROPOUT).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                loss_fn = nn.MSELoss()

                # Train model
                train_rmse, val_rmse_std, val_rmse_orig = train_model(model, optimizer, loss_fn, train_users, train_items, standardized_train_ratings, val_users, val_items, original_val_ratings, standardized_val_ratings, means, stds, EPOCHS, STOP_THRESHOLD, False, hyper_verbose=False)
                # Sum to get cumulative best val loss
                avg_best_val_loss += min(val_rmse_orig)

            # Average losses over splits
            avg_best_val_loss /= splits
            print(f"Runs: {count_runs}/{n_combos}")
            print(f"----- Avg best val loss over splits: {avg_best_val_loss} -----")
                
            # Save results of run
            combo_results = {
                "model": model_name,
                "combo": combo,
                "train_losses": train_rmse,
                "val_losses_std": val_rmse_std,
                "val_losses_orig": val_rmse_orig,
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
    hyperparameter_tuning(SPLITS)