import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, EdgeConv, SAGEConv, GINConv

##############################
#    Basic layers
##############################

def act_layer(act):
    """Returns an activation layer based on the string input."""
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError(f"Activation function {act} is not implemented.")


def norm_layer(norm, nc):
    """Returns a normalization layer."""
    if norm == 'batch':
        return nn.BatchNorm1d(nc, affine=True)
    elif norm == 'instance':
        return nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError(f"Normalization type {norm} is not implemented.")


class DropPath(nn.Module):
    """ Stochastic Depth (DropPath) implementation for regularization """

    def __init__(self, drop_prob=0.0):
        """
        Args:
            drop_prob (float): Probability of dropping paths (0.0 means no drop).
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, ...)
        Returns:
            Tensor: Output tensor with randomly dropped paths during training
        """
        if self.drop_prob == 0.0 or not self.training:
            return x  # No dropout in inference mode or when drop_prob=0

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Shape: (batch_size, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Convert to 0 or 1

        return x.div(keep_prob) * random_tensor  # Scale output to preserve expected value


class MLP(nn.Sequential):
    """Multi-Layer Perceptron"""
    def __init__(self, channels, act='relu', norm=None, bias=True):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1], bias))
            if norm:
                layers.append(norm_layer(norm, channels[i + 1]))
            layers.append(act_layer(act))
        super().__init__(*layers)


class GrapherDGL(nn.Module):
    """
    DGL-based Grapher module using optimized graph convolutions.
    """
    def __init__(self, in_channels, conv='edge', act='relu', norm=None, bias=True, dropout=0.0, drop_path=0.0):
        super(GrapherDGL, self).__init__()

        self.norm = norm_layer(norm, in_channels) if norm else None
        self.act = act_layer(act)

        # Choose DGL's optimized graph convolution
        if conv == 'edge':
            self.conv = EdgeConv(nn.Linear(in_channels * 2, in_channels))
        elif conv == 'sage':
            self.conv = SAGEConv(in_channels, in_channels, aggregator_type="mean")
        elif conv == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels), nn.ReLU(),
                nn.Linear(in_channels, in_channels)
            )
            self.conv = GINConv(mlp, learn_eps=True)
        elif conv == 'gcn':
            self.conv = GraphConv(in_channels, in_channels)
        else:
            raise NotImplementedError(f"Conv type {conv} is not supported.")

        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, g, x):
        x = self.conv(g, x)
        x = self.drop_path(x)

        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DenseDilatedKnnGraphDGL(nn.Module):
    """ 
    Constructs a batch-wise KNN Graph with true dilation for Vision GNN (ViG).
    """
    def __init__(self, k=9, max_dilation=3, dist_metric='euclidean', include_self=False):
        super(DenseDilatedKnnGraphDGL, self).__init__()
        self.k = k
        self.max_dilation = max_dilation
        self.dist_metric = dist_metric
        self.include_self = include_self

    def forward(self, x, layer_idx):
        """
        Args:
            x: (B, N, C) - Batch of feature matrices
            layer_idx: (int) - Current layer index (used to compute dilation)
        Returns:
            Batched DGL Graph with dilated kNN connections
        """
        B, N, C = x.shape

        # Compute dilation factor based on network depth
        dilation = min(layer_idx // 4 + 1, self.max_dilation)
        k_dilated = self.k * dilation

        # Flatten features for DGL's kNN algorithm
        x_flat = x.reshape(-1, C)
        
        # Create segment IDs for batched graph construction
        segs = [N] * B

        # Create full kNN graph with k*dilation neighbors
        g_full = dgl.segmented_knn_graph(
            x_flat, 
            k=k_dilated, 
            segs=segs, 
            algorithm="bruteforce-sharemem", 
        )

        # Extract dilated edges - take every dilation-th edge
        src, dst = g_full.edges()
        src_dilated, dst_dilated = src[::dilation], dst[::dilation]

        # Create new graph with dilated edges
        g_dilated = dgl.graph((src_dilated, dst_dilated), num_nodes=B * N).to(x.device)
        
        # Add self-loops if specified
        if self.include_self:
            g_dilated = dgl.add_self_loop(g_dilated)
            
        return g_dilated