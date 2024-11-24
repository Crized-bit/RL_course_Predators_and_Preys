import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop
from torchrl.modules import NoisyLinear


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return x + y


class ImagePreprocessor(nn.Module):
    def __init__(self, num_input_channels, embedding_size):
        super(ImagePreprocessor, self).__init__()
        # 40 40 2
        self.full_conv = nn.Sequential(
            ConvBlock(in_channels=num_input_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            ResConvBlock(8),
            ResConvBlock(8),
            ResConvBlock(8),
            ConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            ResConvBlock(16),
            ResConvBlock(16),
            ResConvBlock(16),
            ConvBlock(in_channels=16, out_channels=num_input_channels, kernel_size=3, stride=1, padding=1),
            ResConvBlock(num_input_channels),
            ResConvBlock(num_input_channels),
            nn.Flatten(),
            NoisyLinear(40 * 40 * num_input_channels, 256),
        )

        self.size_10_conv = nn.Sequential(
            ResConvBlock(num_input_channels),
            ResConvBlock(num_input_channels),
            ResConvBlock(num_input_channels),
        )
        self.size_10_lin = NoisyLinear(10 * 10 * num_input_channels, 256)

        self.size_5_conv = nn.Sequential(
            ResConvBlock(num_input_channels),
            ResConvBlock(num_input_channels),
        )
        self.size_5_lin = NoisyLinear(5 * 5 * num_input_channels, 256)

        self.linear_last = NoisyLinear(256 * 3, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_5 = self.size_5_lin(self.size_5_conv(center_crop(x, [5, 5])).flatten(start_dim=1))
        x_10 = self.size_10_lin(self.size_10_conv(center_crop(x, [10, 10])).flatten(start_dim=1))
        x_full = self.full_conv(x)
        result = torch.cat((x_full, x_10, x_5), dim=-1)
        return self.linear_last(result)


class RLPreprocessor(nn.Module):
    def __init__(self, num_input_channels, embedding_size):
        super(RLPreprocessor, self).__init__()
        self.ImagePreprocessor = ImagePreprocessor(num_input_channels, embedding_size)
        # self.linear = nn.Linear(256 + 2, 256)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # addtional_features.shape [N_batch, 5, 2]
        # image_features.shape [N_batch, 256] -> [N_batch, 5, 256]
        # N_batch, n_agents, state_dim
        image_features = self.ImagePreprocessor(image)
        return image_features.reshape(-1, 5, image_features.shape[-1])
