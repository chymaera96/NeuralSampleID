import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn import GraphConv, EdgeConv, SAGEConv, GINConv
from encoder.gcn_lib.pos_embed import get_2d_relative_pos_embed
# from pos_embed import get_2d_relative_pos_embed

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



class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (MRConv) for DGL.
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels, bias=bias), 
            norm_layer(norm, out_channels) if norm else nn.Identity(),
            act_layer(act)
        )

    def forward(self, g, x):
        with g.local_scope():
            g.ndata['h'] = x

            g.apply_edges(lambda edges: {'diff': edges.dst['h'] - edges.src['h']})

            g.update_all(
                dgl.function.copy_e('diff', 'm'),
                lambda nodes: {'max_diff': torch.max(nodes.mailbox['m'], dim=1)[0]}
            )

            max_diff = g.ndata['max_diff']
            if max_diff.shape[0] != x.shape[0]:  
                print(f"Shape mismatch: x={x.shape}, max_diff={max_diff.shape}")
                max_diff = torch.zeros_like(x)  

            concat_features = torch.cat([x, max_diff], dim=1)

            return self.nn(concat_features)





class DropPath(nn.Module):
    """ 
    Stochastic Depth (DropPath) implementation matching `timm.models.layers`
    """
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
        
        # Correct boolean mask for dropping paths
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob

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
    def __init__(self, in_channels, conv='edge', act='relu', norm=None, bias=True, dropout=0.0, drop_path=0.0, relative_pos=False, n=196, r=1):

        super(GrapherDGL, self).__init__()

        self.norm = norm_layer(norm, in_channels) if norm else None
        self.act = act_layer(act)

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
        elif conv == 'mr':
            self.conv = MRConv(in_channels, in_channels * 2)
        else:
            raise NotImplementedError(f"Conv type {conv} is not supported.")
        
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels)
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels, int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n // (r*r)), mode='bicubic', align_corners=False
)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def forward(self, g, x):

        B, C, N = x.shape
        _tmp = x

        x = self.act(self.fc1(x))

        x_nodes = x.permute(0, 2, 1).reshape(-1, C)
        x_nodes = self.conv(g, x_nodes)
        x = x_nodes.reshape(B, N,-1).permute(0, 2, 1)

        x = self.fc2(x)

        assert x.shape == _tmp.shape, f"Mismatch in shapes: x={x.shape}, _tmp={_tmp.shape}"


        if self.relative_pos is not None:
            x = x + self.relative_pos.to(x.device)

        x = self.drop_path(x)
        x = x + _tmp

        if self.norm:
            x = self.norm(x)

        x = self.act(x)
        return x


class DenseDilatedKnnGraphDGL(nn.Module):
    """ 
    Constructs a batch-wise KNN Graph with true dilation for Vision GNN (ViG).
    """
    def __init__(self, k=9, max_dilation=3, dist_metric='euclidean', include_self=False, epsilon=0.2):
        """
        Args:
            k (int): Base number of neighbors in kNN graph.
            max_dilation (int): Maximum dilation factor.
            dist_metric (str): Distance metric ('euclidean', 'cosine', etc.).
            include_self (bool): Whether to add self-loops.
            epsilon (float): Probability of randomly dropping edges (stochastic dropout).
        """
        super(DenseDilatedKnnGraphDGL, self).__init__()
        self.k = k
        self.max_dilation = max_dilation
        self.dist_metric = dist_metric
        self.include_self = include_self
        self.epsilon = epsilon  

    def forward(self, x, layer_idx):
        """
        Args:
            x: (B, N, C) - Batch of feature matrices
            layer_idx: (int) - Current layer index (used to compute dilation)
        Returns:
            Batched DGL Graph with dilated kNN connections and stochastic dropout
        """
        B, C, N = x.shape

        # dilation factor
        dilation = min(layer_idx // 4 + 1, self.max_dilation)
        k_dilated = self.k * dilation
        x_flat = x.permute(0,2,1).reshape(-1, C)
        
        # IDs for batched graph construction
        segs = [N] * B

        # kNN graph with k*dilation neighbors
        g_full = dgl.segmented_knn_graph(
            x_flat, 
            k=k_dilated, 
            segs=segs, 
            algorithm="bruteforce-sharemem"
        )

        # Extract dilated edges - take every dilation-th edge
        src, dst = g_full.edges()
        src_dilated, dst_dilated = src[::dilation], dst[::dilation]
        g_dilated = dgl.graph((src_dilated, dst_dilated), num_nodes=B * N).to(x.device)


        if self.training and self.epsilon > 0:
            edge_mask = torch.rand(len(src_dilated), device=x.device) > self.epsilon  # Keep (1 - epsilon) fraction
            src_dilated, dst_dilated = src_dilated[edge_mask], dst_dilated[edge_mask]
            g_dilated = dgl.graph((src_dilated, dst_dilated), num_nodes=B * N).to(x.device)

        # Add self-loops if specified
        if self.include_self:
            g_dilated = dgl.add_self_loop(g_dilated)

        return g_dilated
