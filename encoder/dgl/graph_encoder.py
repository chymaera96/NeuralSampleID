import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv, EdgeConv, SAGEConv, GINConv
from dgl_util import GrapherDGL, DenseDilatedKnnGraphDGL, act_layer, norm_layer



class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.conv(x)

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer(act)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.drop_path(x) + shortcut
        return x

class GraphEncoderDGL(nn.Module):
    def __init__(self, cfg=None, k=3, conv='edge', act='relu', norm='batch', bias=True, dropout=0.0, dilation=True,
                 epsilon=0.2, drop_path=0.1, size='t', emb_dims=1024, in_channels=3):
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
        self.graph_builder = DenseDilatedKnnGraphDGL(k)
        self.stem = nn.Sequential(nn.Conv2d(in_channels,self.channels[0], kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.channels[0]),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(self.blocks))]
        self.backbone = nn.ModuleList([])
        idx = 0
        
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(Downsample(self.channels[i-1], self.channels[i]))
            for j in range(self.blocks[i]):
                self.backbone.append(
                    nn.Sequential(
                        GrapherDGL(self.channels[i], conv=conv, act=act, norm=norm),
                        FFN(self.channels[i], self.channels[i] * 4, act=act, drop_path=dpr[idx])
                    )
                )
                idx += 1
        
        self.proj = nn.Conv2d(self.channels[-1], self.emb_dims, 1, bias=True)

    def forward(self, x):
        x = x.unsqueeze(-1)
        B, C, N, _ = x.shape
        x = self.stem(x)
        g = self.graph_builder(x)
        
        for layer in self.backbone:
            x = layer(g, x)
        
        x = self.proj(x)
        x = torch.mean(x, dim=2).squeeze(-1).squeeze(-1)
        return x
