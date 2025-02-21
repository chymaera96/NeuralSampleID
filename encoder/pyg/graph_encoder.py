import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder.pyg.gcn_lib import Block

class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )


class GraphEncoder(nn.Module):
    def __init__(self, cfg, k=3, dilation=2, size="t", emb_dims=1024, in_channels=3, drop_path=0.1, pre_norm=False):
        super().__init__()
        self.k = k
        self.dilation = dilation
        
        self.blocks, self.channels = {
            "t": ([2, 2, 6, 2], [64, 128, 256, 512]),
            "s": ([2, 2, 6, 2], [80, 160, 400, 640]),
            "m": ([2, 2, 16, 2], [96, 192, 384, 768]),
            "l": ([2, 2, 18, 2], [128, 256, 512, 1024]),
        }[size]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.backbone = nn.ModuleList()
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(self.blocks))]

        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(Downsample(self.channels[i-1], self.channels[i]))
            
            for j in range(self.blocks[i]):
                dilation = min(idx // 4 + 1, 4)
                self.backbone.append(
                    Block(
                        self.channels[i], 
                        k, 
                        dilation=dilation,
                        drop_path=dpr[idx]
                    )
                )
                idx += 1

        self.proj = nn.Conv2d(self.channels[-1], emb_dims, 1, bias=True)
    
    def forward(self, x):
        
        x = x.unsqueeze(-1)
        x = self.stem(x)
        print(f"Shape after stem: {x.shape}")
        
        for block in self.backbone:
            x = block(x)
        
        x = self.proj(x)
        x = torch.mean(x, dim=2).squeeze(-1)
        return x

if __name__ == '__main__':

    encoder = GraphEncoder()
    dummy_tensor = torch.rand(8,3,512)
    out = encoder(dummy_tensor)