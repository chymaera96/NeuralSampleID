import torch
import torch.nn as nn
import torch.nn.functional as F


class GPUPeakExtractorv2(nn.Module):
    def __init__(self, cfg):
        super(GPUPeakExtractorv2, self).__init__()

        self.n_filters = cfg['n_filters']
        self.patch_bins = cfg['patch_bins']
        self.patch_frames = cfg['patch_frames']

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=self.n_filters,
                kernel_size=(self.patch_bins, self.patch_frames),
                stride=(self.patch_bins, self.patch_frames),
            ),
            nn.ReLU(),
        )

        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.n_gpus == 0:
            self.n_gpus = 1

        T_tensor = torch.linspace(0, 1, steps=cfg['n_frames'])
        T_tensor = T_tensor.unsqueeze(0).unsqueeze(1).repeat(cfg['bsz_train'] // self.n_gpus, cfg['n_mels'], 1)
        self.T_tensor = T_tensor

        F_tensor = torch.linspace(0, 1, steps=cfg['n_mels'])
        F_tensor = F_tensor.unsqueeze(0).unsqueeze(2).repeat(cfg['bsz_train'] // self.n_gpus, 1, cfg['n_frames'])
        self.F_tensor = F_tensor
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, spec_tensor):
        min_vals = torch.amin(spec_tensor, dim=(1, 2), keepdim=True)
        max_vals = torch.amax(spec_tensor, dim=(1, 2), keepdim=True)
        spec_tensor = (spec_tensor - min_vals) / (max_vals - min_vals)

        # peaks = self.peak_from_features(spec_tensor.unsqueeze(1))
        peaks = spec_tensor.unsqueeze(1)
        self.T_tensor = self.T_tensor.to(peaks.device)
        self.F_tensor = self.F_tensor.to(peaks.device)

        # Concatenate T_tensor, F_tensor and spec_tensor to get a tensor of shape (batch, 3, H, W)
        try:
            tensor = torch.cat((self.T_tensor.unsqueeze(1), self.F_tensor.unsqueeze(1), peaks), dim=1)
        except RuntimeError as e:
            # Validation case (unused in current setup)
            T_tensor = torch.linspace(0, 1, steps=spec_tensor.shape[2], device=spec_tensor.device)
            T_tensor = T_tensor.unsqueeze(0).unsqueeze(1).repeat(spec_tensor.shape[0], spec_tensor.shape[1], 1)
            F_tensor = torch.linspace(0, 1, steps=spec_tensor.shape[1], device=spec_tensor.device)
            F_tensor = F_tensor.unsqueeze(0).unsqueeze(2).repeat(spec_tensor.shape[0], 1, spec_tensor.shape[2])
            tensor = torch.cat((T_tensor.unsqueeze(1), F_tensor.unsqueeze(1), peaks), dim=1)

        feature = self.convs(tensor)

        B, C, H, W = feature.shape
        feature = feature.reshape(B, C, -1)
        return feature