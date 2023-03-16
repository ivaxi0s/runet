# Code adapted from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsa import RSA


class UNet3D(nn.Module):
    def __init__(self, n_in, n_out,
                 width_multiplier=1,
                 trilinear=False,
                 use_ds_conv=False,
                 use_rsa_enc=False,
                 use_rsa_dec=False,
                 use_rsa_first=False,
                 use_rsa_second=False,
                 use_rsa_pos=None):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_in = number of input channels; 3 for RGB, 1 for grayscale input
          n_out = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if true, DepthwiseSeparableConv3d layers will be used; otherwise, Conv3D.
          use_rsa_conv = if true, RSA layers will used"""

        super(UNet3D, self).__init__()

        if use_rsa_pos is None:
            use_rsa_pos = [1, 2, 3, 4]

        # _channels = (32, 64, 128, 256, 512)
        _channels = (16, 32, 64, 128, 256)
        self.n_in = n_in
        self.n_out = n_out
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(self.n_in, self.channels[0], conv_type=self.convtype)

        use_rsa_enc_first = use_rsa_enc and use_rsa_first
        use_rsa_enc_second = use_rsa_enc and use_rsa_second
        self.down1 = Down(self.channels[0], self.channels[1],
                          conv_type=self.convtype,
                          use_rsa_first=use_rsa_enc_first and (1 in use_rsa_pos),
                          use_rsa_second=use_rsa_enc_second and (1 in use_rsa_pos))
        self.down2 = Down(self.channels[1], self.channels[2],
                          conv_type=self.convtype,
                          use_rsa_first=use_rsa_enc_first and (2 in use_rsa_pos),
                          use_rsa_second=use_rsa_enc and use_rsa_second and (2 in use_rsa_pos))
        self.down3 = Down(self.channels[2], self.channels[3],
                          conv_type=self.convtype,
                          use_rsa_first=use_rsa_enc_first and (3 in use_rsa_pos),
                          use_rsa_second=use_rsa_enc_second and (3 in use_rsa_pos))
        factor = 2 if trilinear else 1
        self.down4 = Down(self.channels[3], self.channels[4] // factor,
                          conv_type=self.convtype,
                          use_rsa_first=use_rsa_enc_first and (4 in use_rsa_pos),
                          use_rsa_second=use_rsa_enc_second and (4 in use_rsa_pos))

        use_rsa_dec_first = use_rsa_dec and use_rsa_first
        use_rsa_dec_second = use_rsa_dec and use_rsa_second
        self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear,
                      use_rsa_first=use_rsa_dec_first and (1 in use_rsa_pos),
                      use_rsa_second=use_rsa_dec_second and (1 in use_rsa_pos))
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear,
                      use_rsa_first=use_rsa_dec_first and (2 in use_rsa_pos),
                      use_rsa_second=use_rsa_dec_second and (2 in use_rsa_pos))
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear,
                      use_rsa_first=use_rsa_dec_first and (3 in use_rsa_pos),
                      use_rsa_second=use_rsa_dec_second and (3 in use_rsa_pos))
        self.up4 = Up(self.channels[1], self.channels[0], trilinear,
                      use_rsa_first=use_rsa_dec_first and (4 in use_rsa_pos),
                      use_rsa_second=use_rsa_dec_second and (4 in use_rsa_pos))
        self.outc = OutConv(self.channels[0], self.n_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None,
                 use_rsa_first=False, use_rsa_second=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        first_conv = conv_type(in_channels, mid_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        second_conv = conv_type(mid_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        if use_rsa_first:
            first_conv = RSA(in_channels, mid_channels, kernel_size=(5, 7, 7), nh=8, dk=0, dv=0, dd=0)
        if use_rsa_second:
            second_conv = RSA(mid_channels, out_channels, kernel_size=(5, 7, 7), nh=8, dk=0, dv=0, dd=0)

        self.double_conv = nn.Sequential(
            first_conv,
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            second_conv,
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, use_rsa_first=True, use_rsa_second=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            DoubleConv(in_channels, out_channels, conv_type=conv_type,
                       use_rsa_first=use_rsa_first, use_rsa_second=use_rsa_second)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, use_rsa_first=True, use_rsa_second=False):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels,
                                   mid_channels=in_channels // 2,
                                   use_rsa_first=use_rsa_first, use_rsa_second=use_rsa_second)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                         kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            self.conv = DoubleConv(in_channels, out_channels,
                                   use_rsa_first=use_rsa_first, use_rsa_second=use_rsa_second)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is SCHW
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_rsa_conv=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        if not use_rsa_conv:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = RSA(in_channels, out_channels, kernel_size=(1, 1, 1),
                            nh=8, dk=0, dv=0, dd=0)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
