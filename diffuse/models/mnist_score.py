from functools import partial

import torch
import torch.nn as nn

from diffuse.models.components.conv_glu import ConvGLU
from diffuse.models.components.unet_parts import DownsampleBlock, UpsampleBlock


class MnistScore(nn.Module):
    def __init__(self, input_dim=1, hidden=64, c_dim=64, nb_timesteps=50):
        """
        A simple U-Net based model, tailored to the dimensions of MNIST.
        Predicts `epsilon` required to invert the forward process.
        (Equivalently, can be considered as the 'score' network in NCSN)
        :param input_dim:
        :param hidden:
        :param c_dim:
        :param nb_timesteps:
        """
        super().__init__()

        # create positional embeddings (Vaswani et al, 2018)
        dims = torch.arange(c_dim // 2).unsqueeze(0)  # (1, c_dim  // 2)
        steps = torch.arange(nb_timesteps).unsqueeze(1)  # (nb_timesteps, 1)
        first_half = torch.sin(steps * 10.0 ** (dims * 4.0 / (c_dim // 2 - 1)))
        second_half = torch.cos(steps * 10.0 ** (dims * 4.0 / (c_dim // 2 - 1)))
        diff_embedding = torch.cat((first_half, second_half), dim=1)  # (nb_timesteps, c_dim)
        self.register_buffer("diff_embedding", diff_embedding)

        # define the main convolution op.
        op = partial(ConvGLU, c_dim=c_dim, kw=3)

        self.init = op(input_dim, hidden)
        self.down1 = DownsampleBlock(hidden, op)  # 14x14
        self.down2 = DownsampleBlock(hidden * 2, op)  # 7x7
        self.up1 = UpsampleBlock(hidden * 4)  # 14x14
        self.up2 = UpsampleBlock(hidden * 2)  # 28x28

        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, x, t):
        """
        Produces an estimate for the noise term `epsilon`.
        :param x: (batch, 1, H, W) torch.float
        :param t: (batch) torch.int
        """

        # get the conditioning for this time-step
        c = self.diff_embedding[t]  # (batch, c_dim)

        # initial channel up-sampling
        x1 = self.init(x, c)
        x2 = self.down1(x1, c)
        x3 = self.down2(x2, c)
        x = self.up1(x3, x2, c)
        x = self.up2(x, x1, c)
        out = self.out(x)

        return out
