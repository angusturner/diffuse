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
        Can equivalently be thought of as a learned gradient or "score function" estimator,
        as part of a Langevin sampling procedure.
        :param x: (batch, 1, H, W) torch.float
        :param t: (batch) torch.int
        """

        # get the conditioning for this time-step
        c = self.diff_embedding[t]  # (batch, c_dim)

        # initial channel up-sampling
        x = self.init(x)
        

        # 28x28
        out, skip = self.l1(out, skip, c)
        out_half, skip_half = self.down1(out, skip)

        # 14x14
        out_half, skip_half = self.l2(out_half, skip_half, c)
        out_quart, skip_quart = self.down2(out_half, skip_half)

        # 7x7
        out_quart, skip_quart = self.l3(out_quart, skip_quart, c)
        out_quart, skip_quart = self.l4(out_quart, skip_quart, c)

        # back to 14x14
        out_half_, skip_half_ = self.up1(out_quart, skip_quart)
        out_half = self.rescale * (out_half + out_half_)
        skip_half = self.rescale * (skip_half + skip_half_)

        # 14x14
        out_half, skip_half = self.l5(out_half, skip_half)

        # back to 28x28
        out_, skip_ = self.up2(out_half, skip_half)
        out = self.rescale * (out + out_)
        skip = self.rescale * (skip + skip_)

        out, skip = self.l6(out, skip, c)
        out = self.out(skip)

        return out
