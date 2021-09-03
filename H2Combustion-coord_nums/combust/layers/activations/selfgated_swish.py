import numpy as np
from torch import nn


def swish(x):
    r"""Compute the self-gated Swish activation function.

    .. math::
       y = x * sigmoid(x)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: Swish activation of input.

    """
    return x * nn.functional.sigmoid(x)