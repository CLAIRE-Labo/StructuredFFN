import torch
import numpy as np


def block_dense_bmm(input, blkdiag, linear):
    batch_shape, h = input.shape[:-1], input.shape[-1]
    batch_dim = np.prod(batch_shape)
    k, q, p = blkdiag.shape
    l, r = linear.shape
    assert k * p == h
    assert r == k * q
    input = input.reshape(batch_dim, k, p).transpose(0, 1)
    out1 = torch.bmm(input, blkdiag.transpose(-1, -2))
    out1 = out1.transpose(0, 1).reshape(batch_dim, r)
    out2 = torch.mm(out1, linear.transpose(-1, -2)).reshape(*batch_shape, l)
    return out2


class BlockDenseCustom(torch.autograd.Function):
    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, w1_bfly, linear):
        # due to bugs in torch.bmm with specific out dtype, we need to change the weight dtype here by hand
        # note that this only changes the weight dtype in this scope
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, r = linear.shape
        assert k * p == n
        assert r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(
            0, 1
        )
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r)
        out2 = torch.mm(out1, linear.transpose(-1, -2)).reshape(*batch_shape, l)
        ctx.save_for_backward(x, w1_bfly, linear, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, linear, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, r = linear.shape

        dx, dw1_bfly, dw2_linear = None, None, None
        dout_reshaped = dout.reshape(batch_dim, l)
        if ctx.needs_input_grad[2]:
            dw2_linear = torch.mm(dout_reshaped.transpose(-1, -2), out1)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = (
                torch.mm(dout_reshaped, linear).reshape(batch_dim, k, q).transpose(0, 1)
            )
            if ctx.needs_input_grad[0]:
                dx = torch.empty(
                    batch_dim, k, p, device=x.device, dtype=x.dtype
                ).transpose(0, 1)
                dx = (
                    torch.bmm(dout1, w1_bfly, out=dx)
                    .transpose(0, 1)
                    .reshape(*batch_shape, n)
                )
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped)
        return dx, dw1_bfly, dw2_linear


block_dense_custom = BlockDenseCustom.apply
