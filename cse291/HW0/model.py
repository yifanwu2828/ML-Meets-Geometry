import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import cse291.commons.pytorch_util as ptu
from icecream import ic


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        '''
        use same conv to keep the orginal size
        bias=False since we are using batch normalization
        '''
        self.double_conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv2d(x)


class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # halve the dimensions 28x28 -> 14x14

        # Down part of UNet
        for feature in features:
            self.downs.append(
                DoubleConv(in_channels, out_channels=feature)
            )
            in_channels = feature

        # Up part of UNet
        for feature in features[::-1]:
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(
                DoubleConv(feature * 2, out_channels=feature)
            )

        self.bottleneck = DoubleConv(
            in_channels=features[-1], out_channels=features[-1] * 2 
        )
        self.final_cov = nn.Conv2d(
            features[0], out_channels=out_channels, kernel_size=1 
        )

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                print("resizing!!")
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat([x, skip_connection], dim=1)
            x = self.ups[i + 1](concat_skip)

        return self.final_cov(x)


def test():
    device = ptu.init_gpu(use_gpu=True, gpu_id=0, verbose=True)
    
    x = torch.randn(1, 3, 572, 572).to(device)
    model = UNET(in_channels=3, out_channels=1).to(device)
    
    pred = model(x)
    ic(x.shape)
    ic(pred.shape)
    # assert x.shape == pred.shape


# if __name__ == "__main__":
    # test()
