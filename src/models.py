"""
The purpose of this script is to define the models used in the project.

The following classes are defined:
    - Model: Base class for all models.
    - BaseLightGCN: Base class for LightGCN models.
    - LightGCNPlus: LightGCN model with additional projections.

The following functions are defined:
    - load_best_val_model: Load the best model from a file.
    - load_model_inputs: Load the model inputs from a file using pickle and reconstruct them.
"""

########## Imports ##########
import pickle
import torch
from torch import nn
from config import N_u, N_v, DEVICE
import torch.nn.functional as F
import numpy as np

########## Models ##########

class Model(nn.Module):
    def __init__(self, ID):
        super().__init__()
        self.ID = ID

    def forward(self, users, items):
        raise NotImplementedError("Derived classes must implement this method")
    
    def get_ratings(self, users, items):
        return self.forward(users, items)

class BaseLightGCN(Model):
    def __init__(self, ID, A_tilde, act_fn, embedding_dim, n_layers, init_emb_std, dropout_rate):
        super().__init__(ID)

        self.A_tilde = A_tilde  # normalized adjacency matrix
        self.K = embedding_dim
        self.L = n_layers 
        self.act_fn = act_fn
        self.init_embs_std = init_emb_std
        self.dropout = dropout_rate

        # Initialize embeddings
        self.E_u = nn.Embedding(num_embeddings=N_u, embedding_dim=self.K)
        self.E_v = nn.Embedding(num_embeddings=N_v, embedding_dim=self.K)
        nn.init.normal_(self.E_u.weight, std=init_emb_std)
        nn.init.normal_(self.E_v.weight, std=init_emb_std)

        # attention

        # Projection to output space after message passing, aggregation, and selection
        self.mlp = self.create_mlp(dropout_rate)

    def create_mlp(self, dropout_rate):
        raise NotImplementedError("Derived classes must implement this method")
    
    def message_passing(self) -> list[torch.Tensor]:
        E_0 = torch.cat([self.E_u.weight, self.E_v.weight], dim=0)  # size (N_u + N_v) x K
        E_layers = [E_0]
        E_l = E_0

        for l in range(self.L):
            E_l = torch.mm(self.A_tilde, E_l)  # shape (N_u + N_v) x K
            E_layers.append(E_l) 
        return E_layers
    
    def aggregate(self, embs: list) -> torch.Tensor:
        """
        Aggregate the embeddings from the message passing layers."""
        
        E_agg = torch.cat(embs, dim=1)
        return E_agg
    
    def select_embeddings(self, users, items, E_agg):
        E_u, E_v = torch.split(E_agg, [N_u, N_v], dim=0)
        # Select embeddings of users and items from the adjacency lists
        E_u = E_u[users]
        E_v = E_v[items]  # shape (N_train, K * (L + 1))
        return E_u, E_v
    
    def forward(self, users, items):
        E_layers = self.message_passing()
        E_agg = self.aggregate(E_layers)
        # input shape for attention: (N_u + N_v) x (K * 2)
        E_u_sel, E_v_sel = self.select_embeddings(users, items, E_agg)

        # Project to output space
        concat_users_items = torch.cat([E_u_sel, E_v_sel], dim=1)  # shape (N_train, 2K * (L + 1))
        out = self.mlp(concat_users_items).squeeze()  
        return out 

class LightGCNPlus(BaseLightGCN):
    def __init__(self, A_tilde, act_fn, embedding_dim, n_layers, init_emb_std, dropout_rate, projections):
        ID = f"{embedding_dim}_{n_layers}_{projections}"
        self.C = projections
        super().__init__(ID, A_tilde, act_fn, embedding_dim, n_layers, init_emb_std, dropout_rate)

    def save_model_inputs(self):
        """
        Save the model inputs to a dictionary using pickle.
        """
        model_inputs = {
            'A_tilde': self.A_tilde,  # Tensor
            'act_fn': self.act_fn,  # Activation function
            'K': self.K,
            'L': self.L,
            'init_embs_std': self.init_embs_std,
            'dropout': self.dropout,
            'projections': self.C
        }
        # Save dictionary to a file using pickle
        with open(f"../data/model_state/model_inputs_{self.ID}.pkl", "wb") as f:
            pickle.dump(model_inputs, f)
    
    def create_mlp(self, dropout_rate):
        layers = []
        input_dim = self.K * 2 * (self.L + 1)
        for proj in self.C:
            output_dim = self.K * proj
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(self.act_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)
    
########## Functions ##########

def load_best_val_model(model_class: Model, ID: str) -> Model:
    """
    Load the best model from a file.
    """
    model = model_class(*load_model_inputs(ID))
    model.load_state_dict(torch.load(f"../data/model_state/best_val_model_{ID}.pth"))
    model = model.to(DEVICE)
    model.eval()
    return model

def load_model_inputs(ID: str) -> tuple:
    """
    Load the model inputs from a file using pickle and reconstruct them.
    """
    filename=f"../data/model_state/model_inputs_{ID}.pkl"
    with open(filename, "rb") as f:
        model_inputs = pickle.load(f)
    
    A_tilde = model_inputs['A_tilde']
    act_fn = model_inputs['act_fn']
    K = model_inputs['K']
    L = model_inputs['L']
    init_embs_std = model_inputs['init_embs_std']
    dropout = model_inputs['dropout']
    projections = model_inputs['projections']
    
    return A_tilde, act_fn, K, L, init_embs_std, dropout, projections