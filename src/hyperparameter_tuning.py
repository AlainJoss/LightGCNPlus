"""
In this script, we define the hyperparameter tuning for the model.
"""

########## Imports ##########
import numpy as np
from itertools import product
from train import train_model
from postprocess import postprocess

########## Functions ##########
def report_run_params(model_name, combo, counter, total_runs) -> None:
    """
    Print the hyperparameters combination.
    """
    print(f"Run {counter}/{total_runs}")
    print(f"Model: {model_name}")
    print(f"HP: L={combo[0]}, K={combo[1]}, INIT_EMBS_STD={combo[2]}, LR={combo[3]}, WEIGHT_DECAY={combo[4]}, DROPOUT={combo[5]}")

def num_combos(num_models, grid) -> None:
    num_combinations = num_models * np.prod([len(v) for v in grid.values()])
    print(f"Number of combinations: {num_combinations}")
    return num_combinations

########## Main ##########

def tune_hyperparameters(train_params: list, grid: dict) -> None:

    model_opt_loss = train_params["model_opt_loss"]
    train_users = train_params["train_users"]
    train_items = train_params["train_items"]
    standardized_train_ratings = train_params["standardized_train_ratings"]
    val_users = train_params["val_users"]
    val_items = train_params["val_items"]
    original_val_ratings = train_params["original_val_ratings"]
    standardized_val_ratings = train_params["standardized_val_ratings"]
    means = train_params["means"]
    stds = train_params["stds"]
    EPOCHS, STOP_THRESHOLD = train_params["EPOCHS"], train_params["STOP_THRESHOLD"]
    
    n_combos = num_combos(len(model_opt_loss[0]), grid)
    counter = 0
    results = {
        "min_val_losses": [],
        "params": []
    }

    for model, optimizer, loss_fn in model_opt_loss:
        for combo in product(*grid.values()):
            counter += 1
            report_run_params(combo, counter, n_combos)
            train_rmse, val_rmse_std, val_rmse_orig = train_model(model, optimizer, loss_fn, train_users, train_items, standardized_train_ratings, val_users, val_items, original_val_ratings, standardized_val_ratings, means, stds, EPOCHS, STOP_THRESHOLD, True, verbosity=50)
            postprocess(model, means, stds)

            results["min_val_losses"].append(min(val_rmse_orig))
            results["params"].append(model.name + ", combo: " + str(combo))

    # Report best hyperparameters
    best_idx = np.argmin(results["min_val_losses"])
    print("Best hyperparameters:")
    print(results["params"][best_idx])
    print("Best val loss:")
    print(results["min_val_losses"][best_idx])

