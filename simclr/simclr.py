import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
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
                                        # nn.Linear(d*u, d)
                                        nn.Linear(d*u, dim)
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
    


class MoCo(nn.Module):
    def __init__(self, cfg, encoder, momentum=0.99):
        super(MoCo, self).__init__()
        self.cfg = cfg
        self.momentum = momentum

        # Attach peak_extractor to encoder
        if cfg['arch'] == 'grafp':
            self.peak_extractor = GPUPeakExtractorv2(cfg)
            self.encoder_q = nn.Sequential(self.peak_extractor, encoder)  # Main encoder (query)
        else:
            self.peak_extractor = None
            self.encoder_q = encoder

        self.add_module("encoder_q", self.encoder_q)

        # Create momentum encoder as a deepcopy of encoder_q
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False  # Freeze momentum encoder

        # Initialize projectors
        d, h, u = cfg['d'], cfg['h'], cfg['u']
        self.projector_q = nn.Sequential(nn.Linear(h, d * u), nn.ELU(), nn.Linear(d * u, d))
        self.projector_k = nn.Sequential(nn.Linear(h, d * u), nn.ELU(), nn.Linear(d * u, d))

        # Copy encoder parameters to momentum encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # Copy weights
            param_k.requires_grad = False  # Freeze momentum encoder

    @torch.no_grad()
    def update_momentum_encoder(self):
        """Momentum update for key encoder (including peak_extractor)."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, x_i, x_j):
        """Forward pass using in-batch negatives instead of a queue."""

        h_i = self.encoder_q(x_i)
        z_i = F.normalize(self.projector_q(h_i), p=2, eps=1e-10)

        with torch.no_grad():
            h_j = self.encoder_k(x_j)  
            z_j = F.normalize(self.projector_k(h_j), p=2, eps=1e-10)

        return h_i, h_j, z_i, z_j