from typing import Tuple

import torch


def unsqueeze_as(x: torch.Tensor, shape: Tuple[..., int]) -> torch.Tensor:
    """
    Add trailing dimensions onto `x` until it matches the rank of a second
    tensor whose shape is given.
    e.g)
    x = torch.randn(3)
    y = torch.randn(3, 4, 1)
    x = unsqueeze_as(x, y.shape) # (3, 1, 1)
    :param x:
    :param shape:
    :return:
    """
    extra_dims = x.dim() - len(shape)
    return x[(...,) + (None,) * extra_dims]
