import torch
import torch.nn as nn


class ConvGLU(nn.Module):
    def __init__(self, channels, c_dim=None, dilation=1, kw=3):
        """
        Convolution with GLU activation and (optional) global conditioning
        :param channels
        :param c_dim:
        :param dilation:
        :param kw:
        """
        super().__init__()

        # TODO: try without batch-norm?
        self.bn = nn.BatchNorm2d(channels)

        # main op. + parameterised residual connection
        padding = (kw - 1) * dilation // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=kw, dilation=dilation, padding=padding)
        self.up1x1 = nn.Conv2d(channels // 2, channels, 1)

        self.rescale = 0.5 ** 0.5

        if c_dim is not None:
            self.c_proj = nn.Linear(c_dim, channels)

    def forward(self, x, c=None):
        """
        :param x: (B, C H, W)
        :param c: optional global conditioning (batch, features)
        """
        h = self.bn(x)
        h = self.conv(h)
        a, b = torch.chunk(h, 2, 1)  # (B, C // 2, H, W)

        if c is not None:
            assert hasattr(self, "c_proj"), "Oops, conditioning dim not specified!"
            batch = x.shape[0]
            c_proj = self.c_proj(c)
            c_a, c_b = torch.chunk(c_proj, 2, -1)  # (B, C // 2)
            c_a = c_a.reshape(batch, -1, 1, 1)  # (B, C // 2, H=1, W=1)
            c_b = c_b.reshape(batch, -1, 1, 1)
            a = (a + c_a) * self.rescale
            b = (b + c_b) * self.rescale

        # main op + residual, and re-scale to preserve variance
        out = torch.sigmoid(a) * b
        out = self.up1x1(out)
        out = self.rescale * (out + x)  # (B, C, H, W)

        return out
