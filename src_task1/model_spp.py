from torch import nn
import torch
import math


class SPPLayer(nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels

    def forward(self, x):
        N, C, H, W = x.size()  # batch_size, num_channels, height, width
        spp = []
        # print(N, C, H, W)
        for i in range(1, self.num_levels + 1):
            layer_size = i

            h = int(math.ceil(H / layer_size))
            w = int(math.ceil(W / layer_size))

            diff_h = abs(h * layer_size - H)
            diff_w = abs(w * layer_size - W)

            h_pad = diff_h // 2
            w_pad = diff_w // 2

            h_left_pad = h_pad
            h_right_pad = diff_h - h_left_pad

            w_left_pad = w_pad
            w_right_pad = diff_w - w_left_pad

            manual_pad = nn.ZeroPad2d((w_left_pad, w_right_pad, h_left_pad, h_right_pad))
            # print(f"layer_size: {layer_size}, h: {h}, w: {w}, h_pad: {h_pad}, w_pad: {w_pad}")
            # print(f"x shape before padding: {x.shape}")
            y = manual_pad(x)
            # print(f"x shape after padding: {y.shape}")            

            # print(f"layer_size: {layer_size}, h: {h}, w: {w}, h_pad: {h_pad}, w_pad: {w_pad}")

            maxpool = nn.MaxPool2d((h, w), stride=(h, w))

            z = maxpool(y)
            z = z.view(N, -1)

            spp.append(z)

        spp = torch.cat(spp, 1)
        # print(f"spp shape: {spp.shape}")
        return spp


class FaceDetector(nn.Module):
    def __init__(self, spp_levels=3):
        super(FaceDetector, self).__init__()
        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Spatial Pyramid Pooling
        self.spp = SPPLayer(num_levels=spp_levels)

        spp_coef = sum([i**2 for i in range(1, spp_levels+1)])

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * spp_coef, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        

    def forward(self, x):
        # Convolutional layers
        x = self.cnn(x)
        # Spatial Pyramid Pooling
        x = self.spp(x)
        # Fully connected layers
        x = self.fc(x)
        return x