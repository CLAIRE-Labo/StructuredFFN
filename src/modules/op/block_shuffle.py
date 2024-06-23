# paste from Monarch by TriDao
import torch
import numpy as np
from einops import rearrange


def block_shuffle_einsum(input, blkdiag1, blkdiag2):
    batch_shape, h = input.shape[:-1], input.shape[-1]
    batch_dim = np.prod(batch_shape)
    k, q, p = blkdiag1.shape
    l, s, r = blkdiag2.shape
    assert k * p == h
    assert l * r == k * q
    input = input.reshape(batch_dim, k, p)
    out1 = torch.einsum("kqp,bkp->bkq", blkdiag1, input)
    out1 = rearrange(rearrange(out1, "b k q -> b (k q)"), "b (r l) -> b l r", l=l)
    return torch.einsum("lsr,blr->bsl", blkdiag2, out1).reshape(*batch_shape, s * l)


def block_shuffle_bmm(input, blkdiag1, blkdiag2):
    batch_shape, h = input.shape[:-1], input.shape[-1]
    batch_dim = np.prod(batch_shape)
    k, q, p = blkdiag1.shape
    l, s, r = blkdiag2.shape
    assert k * p == h
    assert l * r == k * q
    input = input.reshape(batch_dim, k, p).transpose(0, 1)
    out1 = torch.bmm(input, blkdiag1.transpose(-1, -2))
    out1 = (
        out1.transpose(0, 1)
        .reshape(batch_dim, r, l)
        .transpose(-1, -2)
        .contiguous()
        .transpose(0, 1)
    )
    out2 = torch.bmm(out1, blkdiag2.transpose(-1, -2))
    out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
    return out2


class BlockShuffleCustom(torch.autograd.Function):
    # Paste from monarch repo
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
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(
            0, 1
        )
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = (
            out1.transpose(0, 1)
            .reshape(batch_dim, r, l)
            .transpose(-1, -2)
            .contiguous()
            .transpose(0, 1)
        )
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(
            0, 1
        )
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(
                batch_dim, l, r, device=x.device, dtype=x.dtype
            ).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly, out=dout1)
            dout1 = (
                dout1.transpose(0, 1)
                .transpose(-1, -2)
                .contiguous()
                .reshape(batch_dim, k, q)
                .transpose(0, 1)
            )
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = (
                    torch.bmm(dout1, w1_bfly, out=dx.transpose(0, 1))
                    .transpose(0, 1)
                    .reshape(*batch_shape, n)
                )
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped)
        return dx, dw1_bfly, dw2_bfly


block_shuffle_custom = BlockShuffleCustom.apply
