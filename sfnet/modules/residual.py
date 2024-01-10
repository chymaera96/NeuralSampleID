import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualUnit(nn.Module):
    def __init__(self, inplanes, channels, temporal_conv, strides, downsample = None):
        super(ResidualUnit, self).__init__()

        if temporal_conv:
            kernels = [[3,1],[1,3],[1,1]]
        else:
            kernels = [[1,1],[1,3],[1,1]]

        self.conv1 = nn.Sequential(
                        nn.Conv2d(inplanes, channels[1], kernel_size = kernels[0], stride = [1,strides[0]], padding = [int(kernels[0][0] / 2) ,0]),
                        nn.GroupNorm(channels[1], channels[1]),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(channels[1], channels[1], kernel_size = kernels[1], stride = [1,strides[1]], padding = [0,1]),
                        nn.GroupNorm(channels[1], channels[1]),
                        nn.ReLU())
        
        self.conv3 = nn.Sequential(
                        nn.Conv2d(channels[1], channels[2], kernel_size = kernels[2], stride = [1,strides[2]], padding = 0),
                        nn.GroupNorm(channels[2], channels[2]))
        
        self.downsample = downsample
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # print(f" Downsample = {self.downsample}")
        # print(f"Inside res unit: {x.shape}")
        residual = x
        out = self.conv1(x)
        # print(f"After conv1 {out.shape}")
        out = self.conv2(out)
        # print(f"After conv2 {out.shape}")
        out = self.conv3(out)
        # print(f"After conv3 {out.shape}")
        if self.downsample:
            residual = self.downsample(x)
        # print(f"Residual shape {residual.shape}")
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, cfg):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.alpha = cfg['alpha']
        self.beta = cfg['beta']
        self.layers = cfg['layers']
        self.channels = [[64,64,256],[256,128,512],[512,256,1024],[1024,512,2048]] # [dim_in, dim_inner, dim_out]
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size = [1,7], stride = 2, padding = [0,3]), #correct
                        nn.GroupNorm(32,32),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) 
        self.layer2 = self._make_layer(block, self.channels[0], self.layers[0], temporal_conv=False)
        self.layer3 = self._make_layer(block, self.channels[1], self.layers[1], temporal_conv=False)
        self.layer4 = self._make_layer(block, self.channels[2], self.layers[2], temporal_conv=True)
        self.layer5 = self._make_layer(block, self.channels[3], self.layers[3], temporal_conv=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels[3][-1], 1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, channels, blocks, temporal_conv):
        downsample = None
        layers = []
        if temporal_conv:
            inplanes = int(channels[0] + channels[0] / self.alpha)
        else:
            inplanes = channels[0]

        for i in range(blocks):
            if i != 0:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, channels[-1], kernel_size=1, stride=[1,1]),
                    nn.GroupNorm(channels[-1], channels[-1]),
                ) 
            else:
                downsample=None
              
            layers.append(block(inplanes, channels, temporal_conv, downsample))       
            inplanes = channels[-1]

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool
        x = self.layer2(x)
        x = self.layer3(x)  
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x        
    
class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(self, dim_in, alpha, fusion_conv_channel_ratio=2, fusion_kernel=7):

        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv2d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1],
            stride=[alpha, 1],
            padding=[int(fusion_kernel / 2), 0]
        )
        self.bn = nn.GroupNorm(dim_in * fusion_conv_channel_ratio, 
                               dim_in * fusion_conv_channel_ratio)
        self.relu = nn.ReLU()

    def forward(self, x_s, x_f):
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return x_s_fuse, x_f

    

class SlowFastNetwork(nn.Module):
    def __init__(self, block, cfg):
        super(SlowFastNetwork, self).__init__()
        self.slow_inplanes = 64
        self.alpha = cfg['alpha']
        self.beta = cfg['beta']
        self.layers = cfg['layers']
        self.fast_inplanes = self.slow_inplanes / self.beta
        self.channels = [[64,64,256],[256,128,512],[512,256,1024],[1024,512,2048]] # [dim_in, dim_inner, dim_out]
        slow_channels = (torch.Tensor(self.channels)).type(torch.int64).tolist()
        fast_channels = (torch.Tensor(slow_channels) / self.beta).type(torch.int64).tolist()

        self.slow_conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = [1,7], stride = 2, padding = [0,3]), #correct
                        nn.GroupNorm(64, 64),
                        nn.ReLU())
        self.fast_conv1 = nn.Sequential(
                        nn.Conv2d(1, 8, kernel_size = [5,7], stride = 2, padding = [2,3]), #correct
                        nn.GroupNorm(8, 8),
                        nn.ReLU())
                
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) 

        self.slow_layer2 = self._make_layer(block, slow_channels[0], self.layers[0], temporal_conv=False, type="slow", strided=False)
        self.slow_layer3 = self._make_layer(block, slow_channels[1], self.layers[1], temporal_conv=False, type="slow")
        self.slow_layer4 = self._make_layer(block, slow_channels[2], self.layers[2], temporal_conv=True, type="slow")
        self.slow_layer5 = self._make_layer(block, slow_channels[3], self.layers[3], temporal_conv=True, type="slow")

        self.fast_layer2 = self._make_layer(block, fast_channels[0], self.layers[0], temporal_conv=True, type="fast", strided=False)
        self.fast_layer3 = self._make_layer(block, fast_channels[1], self.layers[1], temporal_conv=True, type="fast")
        self.fast_layer4 = self._make_layer(block, fast_channels[2], self.layers[2], temporal_conv=True, type="fast")
        self.fast_layer5 = self._make_layer(block, fast_channels[3], self.layers[3], temporal_conv=True, type="fast")

        self.lateral_conv1 = FuseFastToSlow(dim_in=8, alpha=self.alpha) #correct
        self.lateral_conv2 = FuseFastToSlow(dim_in=32, alpha=self.alpha) #correct ?!
        self.lateral_conv3 = FuseFastToSlow(dim_in=64, alpha=self.alpha) #correct
        self.lateral_conv4 = FuseFastToSlow(dim_in=128, alpha=self.alpha) #correct
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, block, channels, blocks, temporal_conv, type, strided=True):
        downsample = None
        layers = []
        if type == "slow":
            inplanes = int(channels[0] + channels[0] / self.alpha)
        else:
            inplanes = channels[0]

        strides = [1,1,1]
        if strided:
            strides[0] = 2

        for i in range(blocks):

            if strides[0] != 1 or inplanes != channels[-1]:

                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, channels[-1], kernel_size=1, stride=[1,strides[0]]),
                    nn.GroupNorm(channels[-1], channels[-1]),
                ) 
            else:
                downsample=None
              
            layers.append(block(inplanes, channels, temporal_conv, strides, downsample))       
            inplanes = channels[-1]
            strides = [1,1,1]

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x_f = x
        x_s = x[:,:,::self.alpha,:]

        x_s = self.slow_conv1(x_s)
        x_s = self.maxpool(x_s)
        x_f = self.fast_conv1(x_f)
        x_f = self.maxpool(x_f)
        x_s, x_f = self.lateral_conv1(x_s, x_f)

        x_f = self.fast_layer2(x_f)
        x_s = self.slow_layer2(x_s)
        x_s, x_f = self.lateral_conv2(x_s, x_f)

        x_f = self.fast_layer3(x_f)
        x_s = self.slow_layer3(x_s)
        x_s, x_f = self.lateral_conv3(x_s, x_f)

        x_f = self.fast_layer4(x_f)
        x_s = self.slow_layer4(x_s)
        x_s, x_f = self.lateral_conv4(x_s, x_f)

        x_f = self.fast_layer5(x_f)
        x_s = self.slow_layer5(x_s)

        x_s = self.avgpool(x_s)
        x_f = self.avgpool(x_f)
        x = torch.cat([x_s, x_f], 1)
        x = x.view(x.size(0), -1)


        return x

