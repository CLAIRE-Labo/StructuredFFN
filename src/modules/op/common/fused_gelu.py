# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# paste from megatron
import os
import torch
from typing import Callable, Optional, Tuple


jit_fuser = torch.jit.script
if torch.__version__ >= "2" and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1"))):
    jit_fuser = torch.compile


@jit_fuser
def bias_gelu_fused_(inp: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Bias-GeLU fused"""
    x = inp + bias
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@jit_fuser
def gelu_fused_(inp: torch.Tensor) -> torch.Tensor:
    """
    GeLU fused, this is copy of bias_gelu_fused cause jit fusion doesn't allow conditioning.
    """
    x = inp
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@jit_fuser
def dgelu_bgrad_fused_(
    grad_output: torch.Tensor, inp: torch.Tensor, bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bgrad-Dgelu fused"""
    x = inp + bias
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    dgelu = ff * grad_output
    bgrad = dgelu.sum(dim=0)
    return dgelu, bgrad


@jit_fuser
def dgelu_fused_(grad_output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    Dgelu fused, this is copy of bgrad_dgelu_fused_ cause jit fusion doesn't allow conditioning.
    """
    x = inp
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    dgelu = ff * grad_output
    return dgelu


class BiasGeLUFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu_fused_(input, bias)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        return dgelu_bgrad_fused_(grad_output, input, bias)


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fused_(input)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        return dgelu_fused_(grad_output, input)


def bias_gelu_impl(input, bias):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasGeLUFunction.apply(input, bias)
    else:
        output = GeLUFunction.apply(input)
    return (
        output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
    )
