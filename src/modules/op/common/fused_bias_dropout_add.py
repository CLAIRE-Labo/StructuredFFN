# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import os
from typing import Optional, Tuple
import torch


jit_fuser = torch.jit.script
if torch.__version__ >= "2" and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1"))):
    jit_fuser = torch.compile


def _bias_dropout_add_func(x_with_bias, residual, prob, training):
    x, bias = x_with_bias  # unpack

    # If we want to train mixed precision, then the output of this function
    # should be half precision. However, in AMP O1, the input (residual) is
    # in fp32, and it will up-cast the result to fp32, causing pipeline parallel
    # GPU communication to hang. Therefore, we need to cast residual to the same
    # dtype as x.
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)

    if bias is not None:
        x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out
        return out
    else:
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out
        return out


@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, True)


@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, False)


def bias_dropout_add_impl(training):
    if training:
        return bias_dropout_add_fused_train
    else:
        return bias_dropout_add_fused_inference
