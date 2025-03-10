import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv, EdgeConv, SAGEConv, GINConv
from encoder.dgl.dgl_util import GrapherDGL, DenseDilatedKnnGraphDGL, act_layer, norm_layer


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, reduction_ratio=2):
        """
        Args:
            in_dim: Input feature dimension (C)
            out_dim: Output feature dimension (C')
            reduction_ratio: Factor by which nodes (N) are reduced (default: 2)
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=reduction_ratio, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, N) - Batch of graph node features
        Returns:
            x: (B, C', N') - Downsampled features with fewer nodes (N' = N // reduction_ratio)
        """
        x = self.conv(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.act = nn.ReLU() if act == 'relu' else nn.GELU()

        # Keep BatchNorm1d to ensure per-node normalization (no graph mixing)
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.bn1 = nn.BatchNorm1d(hidden_features)

        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        """
        x: (N, C) where N is total nodes across all graphs in the batch
        """
        shortcut = x

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.bn2(x)

        x = self.drop_path(x) + shortcut
        return x


class GraphEncoderDGL(nn.Module):
    def __init__(self, cfg=None, k=3, conv='mr', act='relu', norm='batch', bias=True, dropout=0.0, dilation=True,
                 epsilon=0.2, drop_path=0.1, size='t', emb_dims=1024, in_channels=3, include_self=False):
        super().__init__()

        if size == 't':
            self.blocks = [2, 2, 6, 2]
            self.channels = [64, 128, 256, 512]
        elif size == 's':
            self.blocks = [2, 2, 6, 2]
            self.channels = [80, 160, 400, 640]
        elif size == 'm':
            self.blocks = [2, 2, 16, 2]
            self.channels = [96, 192, 384, 768]
        else:
            self.blocks = [2, 2, 18, 2]
            self.channels = [128, 256, 512, 1024]

        self.k = k
        self.emb_dims = emb_dims

        # Initialize graph builder with configurable self-loops
        self.graph_builder = DenseDilatedKnnGraphDGL(k, include_self=include_self, epsilon=epsilon)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Drop path rate increases with depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(self.blocks))]
        self.backbone = nn.ModuleList([])
        idx = 0

        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(Downsample(self.channels[i-1], self.channels[i]))
            for j in range(self.blocks[i]):
                self.backbone.append(
                    nn.Sequential(
                        GrapherDGL(self.channels[i], conv=conv, act=act, norm=norm, drop_path=dpr[idx]),
                        FFN(self.channels[i], self.channels[i] * 4, act=act, drop_path=dpr[idx])
                    )
                )
                idx += 1

        self.proj = nn.Conv2d(self.channels[-1], self.emb_dims, 1, bias=True)


    def forward(self, x, return_pre_proj=False):
        """
        Args:
            x: Input tensor of shape (B, C, N)

        Returns:
            Output embedding of shape (B, emb_dim)
        """

        x = self.stem(x.unsqueeze(-1)).squeeze(-1)
        B, C, N = x.shape
        x = x.contiguous()

        for layer_idx, block in enumerate(self.backbone):
            if isinstance(block, Downsample):
                x = block(x)
            else:
                x = self._apply_graph_block(x, block, layer_idx)

        x_nodes = x
        # print(f"Before projection: {x.shape}")
        x = self.proj(x.unsqueeze(-1))
        # print(f"After projection: {x.shape}")
        x_emb = x.mean(dim=2).squeeze(-1)
        # print(f"After pooling: {x.shape}")

        if return_pre_proj:
            return x_nodes, x_emb
        else:
            return x_emb

    def _apply_graph_block(self, x, block, layer_idx):
        """Helper method to apply a graph convolution block"""

        graph_conv, ffn = block
        B, C, N = x.shape
        g = self.graph_builder(x, layer_idx)
        x_nodes = graph_conv(g, x)
        x_nodes = ffn(x_nodes)

        # # Reshape to original format
        # x = x_nodes.reshape(B, N, -1).permute(0, 2, 1)

        return x