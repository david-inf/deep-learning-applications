"""Convolutional AutoEncoder"""

from torch import nn


class AutoEncoder(nn.Module):
    """
    Data in range [0,1]
    input_size=[N, 3, 28, 28]
    output_size=[N, 3, 28, 28]
    """

    def __init__(self, num_filters=16):
        super().__init__()
        self.encoder = nn.Sequential(
            downsample(3, num_filters),
            nn.BatchNorm2d(num_filters),
            activation(),
            downsample(num_filters, num_filters*2),
            nn.BatchNorm2d(num_filters*2),
            activation(),
            downsample(num_filters*2, num_filters*4),
            nn.BatchNorm2d(num_filters*4),
            activation(),
        )

        self.decoder = nn.Sequential(
            upsample(num_filters*4, num_filters*8),
            nn.BatchNorm2d(num_filters*8),
            activation(),
            upsample(num_filters*8, num_filters*4),
            nn.BatchNorm2d(num_filters*4),
            activation(),
            upsample(num_filters*4, num_filters*2, stride=1, padding=0),
            nn.BatchNorm2d(num_filters*2),
            activation(),
            upsample(num_filters*2, num_filters),
            nn.BatchNorm2d(num_filters),
            activation(),
            convt1x1(num_filters, 3, padding=1),
            nn.Sigmoid(),  # outputs pixels in [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)  # latent
        decoded = self.decoder(encoded)
        return decoded

    def latent(self, x):
        """Foward pass until latent space"""
        encoded = self.encoder(x)
        return encoded


def downsample(in_channels, out_channels, stride=2, padding=1):
    """Convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=4,
        stride=stride,
        padding=padding,
        bias=False,
    )


def upsample(in_channels, out_channels, stride=2, padding=1):
    """Transposed convolution"""
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=4,
        stride=stride,
        padding=padding,
        bias=False,
    )


def convt1x1(in_channels, out_channels, stride=1, padding=0):
    """Convolution"""
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        bias=False,
    )


def activation():
    """Activation function"""
    return nn.LeakyReLU(
        negative_slope=0.2,
        inplace=True
    )
