# Model State

In order to initialize a twin of the best performing model state (on the validation set) after training, we store the following files in the `model_state` folder:

- model_inputs.pkl: a dictionary containing the model inputs.
In `models.py`:
```python
model_inputs = {
    'A_tilde': A_tilde,  
    'act_fn': act_fn,  
    'K': K,
    'L': L,
    'init_embs_std': init_embs_std,
    'dropout': dropout,
    'projections': projections
}
with open("../data/model_state/model_inputs.pkl", "wb") as f:
    pickle.dump(model_inputs, f)
```
- best_val_model.pth: the model state dictionary containing the best model weights. In `train.py`:
```python
torch.save(model.state_dict(), "../data/model_state/best_val_model.pth")
```

