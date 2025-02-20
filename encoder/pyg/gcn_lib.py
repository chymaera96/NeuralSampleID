import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MRConv
from torch_geometric.utils import knn_graph
from timm.models.layers import DropPath
import numpy as np

def act_layer(act):
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif act == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError(f"Activation '{act}' not implemented")

def norm_layer(norm, channels):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    elif norm == "layer":
        return nn.LayerNorm([channels, 1, 1])
    else:
        raise NotImplementedError(f"Normalization '{norm}' not implemented")

class MLP(nn.Module):
    def __init__(self, channels, act="relu", norm=None, bias=True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
            if norm:
                layers.append(norm_layer(norm, channels[i + 1]))
            if act:
                layers.append(act_layer(act))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act="relu", drop_path=0.0, pre_norm=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.pre_norm = pre_norm
        if self.pre_norm:
            self.norm = nn.LayerNorm(in_features)  # LayerNorm before FFN
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer(act)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_features),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.dropout = nn.Dropout(p=0.1)  # Adding dropout for regularization
    
    def forward(self, x):
        shortcut = x
        if self.pre_norm:
            x = self.norm(x)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class MRConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k, dilation=1, relative_pos=True, learnable_rel_pos=True):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.relative_pos = relative_pos
        self.learnable_rel_pos = learnable_rel_pos
        self.conv = MRConv(in_channels, out_channels)
        
        if self.relative_pos and self.learnable_rel_pos:
            self.rel_pos_embed = nn.Embedding(num_embeddings=(2 * k - 1) ** 2, embedding_dim=in_channels)

    def forward(self, x):
        edge_index = dilated_knn_graph(x, self.k, dilation=self.dilation)
        
        if self.relative_pos:
            if self.learnable_rel_pos:
                rel_pos = self.compute_relative_positions(x.shape[1])
                edge_features = self.rel_pos_embed(rel_pos.to(x.device))
                x = x + edge_features
            else:
                rel_pos = torch.from_numpy(get_2d_relative_pos_embed(x.shape[1], int(x.shape[1] ** 0.5))).to(x.device)
                x = x + rel_pos
        
        return self.conv(x, edge_index)

    def compute_relative_positions(self, grid_size):
        indices = torch.arange(grid_size)
        rel_indices = indices.view(-1, 1) - indices.view(1, -1)
        rel_indices += (grid_size - 1)
        return rel_indices.view(-1)
    
class Block(nn.Module):
    def __init__(self, dim, k, dilation, drop_path=0., pre_norm=False):
        super().__init__()
        self.mrconv = MRConvLayer(dim, dim, k, dilation=dilation)
        self.ffn = FFN(dim, hidden_features=dim * 4, drop_path=drop_path, pre_norm=pre_norm)

    def forward(self, x):
        shortcut = x
        x = self.mrconv(x)
        x = self.ffn(x)
        return x + shortcut

def dilated_knn_graph(x, k=9, dilation=1):
    edge_index = knn_graph(x, k * dilation, loop=False)
    edge_index = edge_index[:, ::dilation]
    return edge_index

def get_2d_relative_pos_embed(embed_dim, grid_size):
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    return 2 * np.matmul(pos_embed, pos_embed.T) / pos_embed.shape[1]

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = 1.0 / (10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2)))
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
