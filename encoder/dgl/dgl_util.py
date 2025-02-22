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
        return nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        return nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(f"Normalization type {norm} is not implemented.")


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

    def __init__(self, in_channels, conv='edge', act='relu', norm=None, bias=True, dropout=0.0):
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

    def forward(self, g, x):
        x = self.conv(g, x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DenseDilatedKnnGraphDGL(nn.Module):
    """ Constructs a batch-wise KNN Graph using segmented_knn_graph. """

    def __init__(self, k=9):
        super(DenseDilatedKnnGraphDGL, self).__init__()
        self.k = k

    def forward(self, x):
        """
        Args:
            x: (B, N, C) - Batch of feature matrices
        Returns:
            Batched DGL Graph
        """
        B, N, C = x.shape  # Confirm shape

        # Flatten features into (B*N, C)
        x_flat = x.reshape(-1, C)  # Shape: (B*N, C)

        # Generate segment IDs 
        segs = [N] * B

        print(f"Using k = {self.k}")  # Check if k is correctly set

        # Create graph using segmented_knn_graph
        g = dgl.segmented_knn_graph(x_flat, k=self.k, segs=segs, algorithm="bruteforce-sharemem")  # Move back to CUDA

        print(f"Graph number of nodes after batch correction: {g.num_nodes()}")  # Should be 4096
        return g



