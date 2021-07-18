import torch
import torch.nn as nn

from diffuse.models.components.conv_glu import ConvGLU


class DownsampleBlock(nn.Module):
    def __init__(self, hidden, main_op=ConvGLU):
        """
        Halve the spatial dimensions and double the channels
        :param hidden:
        """
        super(DownsampleBlock, self).__init__()

        self.down = nn.Conv2d(hidden, hidden * 2, kernel_size=2, stride=2)
        self.conv = main_op(hidden, hidden * 2)

    def forward(self, x, c=None):
        down = self.down(x)
        return self.conv(down, c)


class UpsampleBlock(nn.Module):
    def __init__(self, hidden, main_op=ConvGLU):
        super(UpsampleBlock, self).__init__()

        self.up = nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=2, stride=2)
        self.main = main_op(hidden, hidden)

    def forward(self, x1, x2, c=None):
        x1 = self.up(x1)
        feats = torch.cat((x1, x2), dim=1)
        return self.main(feats, c)
