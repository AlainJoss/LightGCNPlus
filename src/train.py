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

    return J.item()

def evaluate_one_epoch(model, loss_fn, users, items, ratings) -> float:
    """
    Evaluate the model for one epoch.
    """
    model.eval()
    with torch.no_grad():
        preds = model.forward(users, items)
        J = loss_fn(preds, ratings)

    return J.item()

def evaluate_one_epoch_original(model, loss_fn, users, items, ratings, means, stds) -> float:
    """
    Evaluate the model for one epoch.
    """
    model.eval()
    with torch.no_grad():
        preds = model.forward(users, items)
        reversed_preds = reverse_standardization(preds, means, stds, users, items)
        J = loss_fn(reversed_preds, ratings)
    return J.item()

def reverse_standardization(preds, means, stds, users, items) -> torch.Tensor:
    pred_rating_matrix = np.zeros((N_u, N_v))
    pred_rating_matrix[users.cpu().numpy(), items.cpu().numpy()] = preds.cpu().numpy()
    reversed_ratings = pred_rating_matrix * stds + means
    reversed_ratings = reversed_ratings[users.cpu().numpy(), items.cpu().numpy()]
    reversed_ratings = torch.tensor(reversed_ratings, dtype=torch.float32, device=DEVICE)
    return reversed_ratings

def save_model_on_val_improvement(model, best_loss, last_loss):
    """
    Save the model if the validation loss has improved.
    """
    if last_loss < best_loss:
        best_loss = last_loss
        torch.save(model.state_dict(), "../data/model_state/best_val_model.pth")

def report_losses(epoch, train_loss, val_loss_standardized, val_loss_original, verbosity):
    """
    Print the training and validation losses.
    """
    if epoch % verbosity == 0:
        print(f"Epoch {epoch} - Train loss: {train_loss:.4f} - Val loss: {val_loss_standardized:.4f} - Val loss original: {val_loss_original:.4f}")

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

def train_model(model, optimizer, loss_fn, train_users, train_items, standardized_train_ratings, val_users, val_items, orig_val_ratings, standardized_val_ratings, means, stds, n_epochs, stop_threshold, save_best_model, verbosity=1) -> tuple[list, list]:
    """
    Train the model.
    """
    best_loss = float('inf')
    train_losses = []
    val_losses_std = []
    val_losses_orig = []
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_users, train_items, standardized_train_ratings)
        val_loss_standardized = evaluate_one_epoch(model, loss_fn, val_users, val_items, standardized_val_ratings)
        val_loss_original = evaluate_one_epoch_original(model, loss_fn, val_users, val_items, orig_val_ratings, means, stds)
        report_losses(epoch, train_loss, val_loss_standardized, val_loss_original, verbosity)
        if save_best_model:
            save_model_on_val_improvement(model, best_loss, val_loss_standardized)
        train_losses.append(train_loss)
        val_losses_std.append(val_loss_standardized)
        val_losses_orig.append(val_loss_original)
        if early_stopping(epoch, train_losses, stop_threshold):
            break
    report_best_val_loss(val_losses_orig)

    return train_losses, val_losses_std, val_losses_orig