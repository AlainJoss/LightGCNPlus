"""
In this script, we will define the models that we will use to predict the movie ratings.
We define the following models:
    - ConcatNonLinear: a GCN that concatenates the embeddings after message passing and applies a non-linear transformation to predict the ratings.
    - ...
"""

########## Imports ##########
import pickle
import torch
from torch import nn
from config import N_u, N_v, DEVICE
import torch.nn.functional as F

########## Functions ##########

def save_model_inputs(A_tilde, act_fn, K, L, init_embs_std, dropout, projections):
    """
    Save the model inputs to a dictionary using pickle.
    """
    model_inputs = {
        'A_tilde': A_tilde,  # Tensor
        'act_fn': act_fn,  # Activation function
        'K': K,
        'L': L,
        'init_embs_std': init_embs_std,
        'dropout': dropout,
        'projections': projections
    }
    # Save dictionary to a file using pickle
    with open("../data/model_state/model_inputs.pkl", "wb") as f:
        pickle.dump(model_inputs, f)

def load_best_val_model():
    """
    Load the best model from a file.
    """
    model = LightGCN(*load_model_inputs())
    model.load_state_dict(torch.load("../data/model_state/best_val_model.pth"))
    return model.to(DEVICE)

def load_model_inputs(filename="../data/model_state/model_inputs.pkl"):
    """
    Load the model inputs from a file using pickle and reconstruct them.
    """
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

########## Models ##########

class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.name = model_name
    
    def forward(self, users, items):
        raise NotImplementedError("Derived classes must implement this method")
    
    def get_ratings(self, users, items):
        raise NotImplementedError("Derived classes must implement this method")

class BaseLightGCN(Model):
    def __init__(self, model_name, A_tilde, act_fn, embedding_dim, n_layers, init_emb_std, dropout_rate):
        super(BaseLightGCN, self).__init__(model_name)

        self.A_tilde = A_tilde  # normalized adjacency matrix
        self.K = embedding_dim
        self.L = n_layers 
        self.act_fn = act_fn

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

    def get_ratings(self, users, items):
        return self.forward(users, items)

class LightGCN(BaseLightGCN):
    def __init__(self, A_tilde, act_fn, embedding_dim, n_layers, init_emb_std, dropout_rate, projections, model_name=""):
        self.projections = projections
        super().__init__(model_name, A_tilde, act_fn, embedding_dim, n_layers, init_emb_std, dropout_rate)

        # For reproducibility after training
        save_model_inputs(A_tilde, act_fn, embedding_dim, n_layers, init_emb_std, dropout_rate, projections)

    def create_mlp(self, dropout_rate):
        layers = []
        input_dim = self.K * 2 * (self.L + 1)
        for proj in self.projections:
            output_dim = self.K * proj
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(self.act_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)