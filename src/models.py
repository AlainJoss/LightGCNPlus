"""
In this script, we will define the models that we will use to predict the movie ratings.
We define the following models:
    - ConcatNonLinear: a GCN that concatenates the embeddings after message passing and applies a non-linear transformation to predict the ratings.
    - ...
"""


########## Imports ##########
import torch
from torch import nn
import torch.nn.functional as F

from config import N_u, N_v


########## Models ##########


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout
        
        # Weight matrix for the linear transformation
        self.W = nn.Linear(in_features, out_features)

        # Attention mechanism weights
        self.a1 = nn.Linear(out_features, 1)
        self.a2 = nn.Linear(out_features, 1)

        # LeakyReLU activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _attention_scores(self, Wh: torch.Tensor):
        
        Wha1 = self.a1(Wh)
        Wha2 = self.a2(Wh)

        e = Wha1 + Wha2.t()
                
        return self.leakyrelu(e)
    
    def forward(self, h, adj):
        Wh = self.W(h)
        Wh = F.dropout(Wh, self.dropout, training=self.training)
        
        e = self._attention_scores(Wh)

        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)        
        e = torch.where(adj > 0, e, zero_vec)

        # Softmax to normalize attention coefficients
        attention = F.softmax(e, dim=1)

        # Apply dropout to attention coefficients
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Linear combination of the features with the attention coefficients
        h_prime = torch.matmul(attention, Wh)
        return h_prime

class MultiGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, alpha=0.2, dropout=0.6, concat=True):
        super(MultiGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.concat = concat

        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, alpha, dropout) for _ in range(num_heads)
        ])
        
        if concat:
            self.out_features = out_features * self.num_heads
        else:
            self.out_features = out_features
        


    def forward(self, h, adj):
        head_outputs = []
        for head in self.attention_heads:
            out = head(h, adj)
            head_outputs.append(out)

        if self.concat:
            # Concatenate the outputs of each head
            h_prime = torch.cat(head_outputs, dim=1)
        else:
            # Average the outputs of each head (used in the final layer)
            h_prime = torch.mean(torch.stack(head_outputs), dim=0)
        
        return h_prime


class ConcatNonLinear(nn.Module):
    def __init__(self, A_tilde, embedding_dim, n_layers, init_emb_std, dropout_rate=0.2, num_heads=1):
        super(ConcatNonLinear, self).__init__()
        self.A_tilde = A_tilde  # normalized adjacency matrix
        self.K = embedding_dim
        self.L = n_layers 
        self.num_heads = num_heads

        # Initialize embeddings
        self.E_u = nn.Embedding(num_embeddings=N_u, embedding_dim=self.K)
        self.E_v = nn.Embedding(num_embeddings=N_v, embedding_dim=self.K)
        nn.init.normal_(self.E_u.weight, std=init_emb_std)
        nn.init.normal_(self.E_v.weight, std=init_emb_std)

        # attention
        self.attention = MultiGraphAttentionLayer(in_features=self.K * (self.L+1), out_features=self.K * (self.L+1), num_heads=self.num_heads, concat=False)

        # Projection to output space after message passing, aggregation, and selection
        self.mlp = nn.Sequential(
            nn.Linear(self.K * 2 * (self.L + 1), 2*self.K),  # if L=1, the linear projects from 4K to 2K
            nn.GELU(),         
            nn.Dropout(dropout_rate),
            nn.Linear(2*self.K, self.K),              
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.K, 1)
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
        E_aggregated = self.aggregate(E_layers)
        adj = torch.zeros(size=(N_u + N_v, N_u + N_v), device=users.device)
        adj[items+N_u, users] = 1
        adj[users, items+N_u] = 1
        
        E_attention = self.attention(E_aggregated, adj)

        E_u_sel, E_v_sel = self.select_embeddings(users, items, E_attention)

        # Project to output space
        concat_users_items = torch.cat([E_u_sel, E_v_sel], dim=1)  # shape (N_train, 2K * (L + 1))
        out = self.mlp(concat_users_items).squeeze()  
        return out 
    
    def get_ratings(self, users, items):
        return self.forward(users, items)
    
    

class ConcatNonLinear_41out(nn.Module):
    def __init__(self, A_tilde, embedding_dim, n_layers, init_emb_std, dropout_rate=0.2):
        super(ConcatNonLinear_41out, self).__init__()
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
            nn.Linear(self.K * 2 * (self.L + 1), self.K), 
            nn.GELU(),          
            nn.Dropout(dropout_rate),
            nn.Linear(self.K, 1)          
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
    

class ConcatNonLinear_42out(nn.Module):
    def __init__(self, A_tilde, embedding_dim, n_layers, init_emb_std, dropout_rate=0.2):
        super(ConcatNonLinear_42out, self).__init__()
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
            nn.Linear(self.K * 2 * (self.L + 1), 2*self.K), 
            nn.GELU(),          
            nn.Dropout(dropout_rate),
            nn.Linear(2*self.K, 1)          
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
    

class ConcatNonLinear421out(nn.Module):
    def __init__(self, A_tilde, embedding_dim, n_layers, init_emb_std, dropout_rate=0.2):
        super(ConcatNonLinear421out, self).__init__()
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
            nn.Linear(self.K * 2 * (self.L + 1), 2*self.K),  # if L=1, the linear projects from 4K to 2K
            nn.GELU(),         
            nn.Dropout(dropout_rate),
            nn.Linear(2*self.K, self.K),              
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.K, 1)
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
    
    
    
    
    