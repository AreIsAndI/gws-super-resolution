# necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid

Tensor = torch.Tensor

# create the necessary classes
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=14):
        super(GeneratorResNet, self).__init__()

        # first layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9,
                                             stride=1, padding=4), nn.PReLU())

        # residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,
                                             stride=1, padding=1),
                                   nn.BatchNorm2d(64, 0.8))

        # upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9,
                                             stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

# class for new image
class NewImage(Dataset):
    def __init__(self, pil_image, img_shape):
        img_height, img_width = img_shape
        # Transforms images
        self.shape_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.445, 0.445, 0.445], # mean
                                     [0.269, 0.269, 0.269]), # standard deviation
            ]
        )
        self.pil_image = pil_image

    def tensorItem(self):
        transformed_img = self.shape_transform(self.pil_image)

        return transformed_img