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
from config import MODEL_PATH


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

def save_model_on_val_improvement(model, optimizer, best_loss, last_loss):
    """
    Save the model if the validation loss has improved.
    """
    if last_loss < best_loss:
        best_loss = last_loss
        torch.save(model.state_dict(), "../data/logs/best_val_model.pth")

def report_losses(epoch, train_loss, val_loss, hyper_verbose):
    """
    Print the training and validation losses.
    """
    if hyper_verbose:
        print(f"Epoch {epoch} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
    else:
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

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
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_val_epoch}")


########## Main ##########

def train_model(model, optimizer, loss_fn, train_users, train_items, train_ratings, val_users, val_items, val_ratings, n_epochs, stop_threshold, save_best_model, hyper_verbose=True) -> tuple[list, list]:
    """
    Train the model.
    """
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_users, train_items, train_ratings)
        val_loss = evaluate_one_epoch(model, loss_fn, val_users, val_items, val_ratings)
        report_losses(epoch, train_loss, val_loss, hyper_verbose)
        if save_best_model:
            save_model_on_val_improvement(model, optimizer, best_loss, val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if early_stopping(epoch, train_losses, stop_threshold):
            break
    report_best_val_loss(val_losses)

    return train_losses, val_losses