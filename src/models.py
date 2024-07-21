"""
In this script, we will define the models that we will use to predict the movie ratings.
We define the following models:
    - ConcatNonLinear: a GCN that concatenates the embeddings after message passing and applies a non-linear transformation to predict the ratings.
    - ...
"""


########## Imports ##########
import torch
from torch import nn
from config import N_u, N_v


########## Models ##########

class ConcatNonLinear(nn.Module):
    def __init__(self, A_tilde, embedding_dim, n_layers, init_emb_std, dropout_rate=0.2):
        super(ConcatNonLinear, self).__init__()
        self.A_tilde = A_tilde  # normalized adjacency matrix
        self.K = embedding_dim
        self.L = n_layers 

        # Initialize embeddings
        self.E_u = nn.Embedding(num_embeddings=N_u, embedding_dim=self.K)
        self.E_v = nn.Embedding(num_embeddings=N_v, embedding_dim=self.K)
        nn.init.normal_(self.E_u.weight, std=init_emb_std)
        nn.init.normal_(self.E_v.weight, std=init_emb_std)

        # attention

        # Projection to output space after message passing, aggregation, and selection
        self.mlp = nn.Sequential(
            nn.Linear(self.K * 2 * (self.L + 1), self.K),  # if L=1, the linear projects from 4K to 2K
            nn.GELU(),          # try 4K->K->1, 4K->2K->1, 4K->2K->K->1
            nn.Dropout(dropout_rate),  # Adding dropout after the activation function
            nn.Linear(self.K, 1)  #,                
            # nn.GELU(),
            # nn.Linear(self.K, 1)
        )

    def message_passing(self) -> torch.Tensor:
        E_0 = torch.cat([self.E_u.weight, self.E_v.weight], dim=0)  # size (N_u + N_v) x K
        E_layers = [E_0]
        E_l = E_0

        for l in range(self.L):
            E_l = torch.mm(self.A_tilde, E_l)  # shape (N_u + N_v) x K
            E_layers.append(E_l) 
        return E_layers
    
    def aggregate(self, embs: list) -> torch.Tensor:
        E_combined = torch.cat(embs, dim=1)
        return E_combined
    
    def select_embeddings(self, users, items, E_combined):
        E_u, E_v = torch.split(E_combined, [N_u, N_v], dim=0)
        # Select embeddings for users and items (aligned)
        E_u = E_u[users]
        E_v = E_v[items]  # shape (N_train, K * (L + 1))
        return E_u, E_v
    
    def forward(self, users, items):
        E_layers = self.message_passing()
        # attention
        E_aggregated = self.aggregate(E_layers)
        E_u_sel, E_v_sel = self.select_embeddings(users, items, E_aggregated)

        # Project to output space
        concat_users_items = torch.cat([E_u_sel, E_v_sel], dim=1)  # shape (N_train, 2K * (L + 1))
        out = self.mlp(concat_users_items).squeeze()  
        return out 
    
    def get_ratings(self, users, items):
        return self.forward(users, items)