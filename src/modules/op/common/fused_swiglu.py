# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# paste from megatron
import torch
import torch.nn.functional as F
import os

jit_fuser = torch.jit.script
if torch.__version__ >= "2" and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1"))):
    jit_fuser = torch.compile


@jit_fuser
def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y, bias):
    y = y + bias
    return swiglu(y)


@jit_fuser
def swiglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (
            g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2,
            g * F.silu(y_1),
        ),
        -1,
    )


@jit_fuser
def bias_swiglu_back(g, y, bias):
    y = y + bias
    dy = swiglu_back(g, y)
    bgrad = dy.sum(dim=0)
    return dy, bgrad


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_swiglu(input, bias)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        return bias_swiglu_back(grad_output, input, bias)


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return swiglu(input)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        return swiglu_back(grad_output, input)


def bias_swiglu_impl(input, bias):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(input, bias)
    else:
        output = SwiGLUFunction.apply(input)

    return (
        output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
    )
