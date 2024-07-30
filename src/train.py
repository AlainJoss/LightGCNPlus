"""
In this script, we will define the training loop for the models that we defined in models.py.
We will define the following functions:
    - train: the training loop for the model.
    - evaluate: the evaluation loop for the model.
    - predict: the prediction loop for the model.
    - save_model: save the model to a file.
    - load_model: load the model from a file.
"""

########## Imports ##########
import torch
import numpy as np
from config import DEVICE
from config import N_u, N_v

########## Functions ##########

def train_one_epoch(model, optimizer, loss_fn, users, items, ratings) -> float:
    """
    Train the model for one epoch.
    """
    model.train()
    optimizer.zero_grad()
    preds = model.forward(users, items)
    J = loss_fn(preds, ratings)
    J.backward()
    optimizer.step()
    return torch.sqrt(J).item()

def evaluate_one_epoch(model, loss_fn, users, items, ratings) -> float:
    """
    Evaluate the model for one epoch.
    """
    model.eval()
    with torch.no_grad():
        preds = model.forward(users, items)
        J = loss_fn(preds, ratings)
    return torch.sqrt(J).item()

def save_model_on_val_improvement(model, best_loss, current_loss):
    """
    Save the model if the validation loss has improved.
    """
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save(model.state_dict(), f"../data/model_state/best_val_model_{model.ID}.pth")

def report_losses(epoch, train_losses, val_losses, best_loss, verbosity):
    """
    Print the training and validation losses.
    """
    if epoch % verbosity == 0 and epoch > 0:
        moving_avg_train = np.mean(train_losses[-verbosity:])
        moving_avg_val = np.mean(val_losses[-verbosity:])
        print(f"Epoch {epoch} - Best Val: {best_loss:.4f} at {val_losses.index(best_loss) + 1} - mv-avg: - Train: {moving_avg_train:.4f} - Val: {moving_avg_val:.4f}")

def early_stopping(epoch, train_losses, stop_threshold) -> bool:
    """
    Check if the model should stop training early.
    """
    if epoch > 0 \
        and train_losses[-2] - train_losses[-1] > 0 \
        and abs(train_losses[-2] - train_losses[-1]) < stop_threshold:
        return True
    return False

def report_best_val_loss(val_losses) -> None:
    """
    Report the best validation loss and the epoch at which it was achieved.
    """
    best_val_loss = min(val_losses)
    best_val_epoch = val_losses.index(best_val_loss)
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_val_epoch + 1}")

########## Main ##########

def train_model(model, optimizer, loss_fn, train_users, train_items, train_ratings, val_users, val_items, val_ratings, n_epochs, stop_threshold, save_best_model=False, verbosity=1) -> tuple[list, list]:
    """
    Train the model.
    Note: trains on MSE, evaluates on RMSE.
    """
    if save_best_model:
        model.save_model_inputs()
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_users, train_items, train_ratings)
        val_loss = evaluate_one_epoch(model, loss_fn, val_users, val_items, val_ratings)
        best_loss = min(val_loss, best_loss)
        if save_best_model:
            save_model_on_val_improvement(model, best_loss, val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        report_losses(epoch, train_losses, val_losses, best_loss,  verbosity)
        if early_stopping(epoch, train_losses, stop_threshold):
            break
    report_best_val_loss(val_losses)

    return train_losses, val_losses