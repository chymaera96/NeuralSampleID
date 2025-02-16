import torch
import torch.nn as nn
import torch.nn.functional as F
from peak_extractor import GPUPeakExtractorv2


class SimCLR(nn.Module):
    def __init__(self, cfg, encoder):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.cfg = cfg
        
        d = cfg['d']
        h = cfg['h']
        u = cfg['u']
        dim = cfg['dim']
        if cfg['arch'] == 'grafp':
            self.peak_extractor = GPUPeakExtractorv2(cfg)
        else:
            self.peak_extractor = None

        if self.cfg['arch'] == 'resnet-ibn':
            self.projector = nn.Identity()
        elif self.cfg['arch'] == 'grafp':              
            self.projector = nn.Sequential(nn.Linear(h, d*u),
                                        nn.ELU(),
                                        nn.Linear(d*u, d)
                                        # nn.Linear(d*u, dim)
                               )

    def forward(self, x_i, x_j):
        
        if self.cfg['arch'] == 'grafp':
            x_i = self.peak_extractor(x_i)

        h_i = self.encoder(x_i)
        z_i = self.projector(h_i)
        z_i = F.normalize(z_i, p=2, eps=1e-10)

        if self.cfg['arch'] == 'grafp':
            x_j = self.peak_extractor(x_j)
        h_j = self.encoder(x_j)
        z_j = self.projector(h_j)
        z_j = F.normalize(z_j, p=2, eps=1e-10)


        return h_i, h_j, z_i, z_j