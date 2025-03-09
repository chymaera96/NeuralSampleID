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
    def __init__(self, cfg, encoder, queue_size=65536, momentum=0.99):
        super(MoCo, self).__init__()
        self.cfg = cfg
        self.momentum = momentum

        # Attach peak_extractor to encoder
        if cfg['arch'] == 'grafp':
            self.peak_extractor = GPUPeakExtractorv2(cfg)
            self.encoder_q = nn.Sequential(self.peak_extractor, encoder)  # Ensure peak_extractor is inside encoder
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

        # Queue for negatives
        self.queue_size = queue_size
        self.register_buffer("queue", torch.randn(d, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Copy encoder parameters to momentum encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # Copy weights
            param_k.requires_grad = False  # Freeze momentum encoder

    @torch.no_grad()
    def update_momentum_encoder(self):
        """Momentum update for key encoder (includes peak_extractor)."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        """Enqueue new keys and dequeue the oldest ones."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_i, x_j):
        """Forward pass with shape check."""

        # Apply peak_extractor if needed
        if self.cfg['arch'] == 'grafp':
            x_i = self.peak_extractor(x_i)
            x_j = self.peak_extractor(x_j)

            # **Shape Mismatch Check**
            expected_channels = self.encoder_q[1].in_channels if isinstance(self.encoder_q, nn.Sequential) else self.encoder_q.in_channels
            if x_i.shape[1] != expected_channels:
                raise RuntimeError(f"Shape mismatch: peak_extractor output has {x_i.shape[1]} channels, but encoder expects {expected_channels}.")

        # Encode queries
        h_i = self.encoder_q(x_i)
        h_j = self.encoder_q(x_j)

        z_i = F.normalize(self.projector_q(h_i), p=2, eps=1e-10)
        z_j = F.normalize(self.projector_q(h_j), p=2, eps=1e-10)

        # Encode keys with momentum encoder (detached)
        with torch.no_grad():
            self.update_momentum_encoder()
            k_i = F.normalize(self.projector_k(self.encoder_k(x_i)), p=2, eps=1e-10)
            k_j = F.normalize(self.projector_k(self.encoder_k(x_j)), p=2, eps=1e-10)

        # Update queue
        self.enqueue_dequeue(k_i)
        self.enqueue_dequeue(k_j)

        return z_i, z_j, k_i, k_j
