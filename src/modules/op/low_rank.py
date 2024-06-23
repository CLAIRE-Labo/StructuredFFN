import torch
import numpy as np


def low_rank_custom(input, linear1, linear2):
    batch_shape, h = input.shape[:-1], input.shape[-1]
    batch_dim = np.prod(batch_shape)
    input = input.reshape(batch_dim, h)
    out2 = torch.mm(
        torch.mm(input, linear1.transpose(-1, -2)), linear2.transpose(-1, -2)
    ).reshape(*batch_shape, -1)
    return out2
