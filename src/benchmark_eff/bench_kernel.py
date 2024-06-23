import os
import sys
import triton
import torch

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

print(project_root)
sys.path.insert(0, project_root)

from src.modules.op import (
    block_shuffle_bmm,
    block_shuffle_einsum,
    block_shuffle_custom,
    block_dense_bmm,
    block_dense_custom,
    low_rank_custom,
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "blocks", "in_blksz", "out_blksz"],
        x_vals=[
            (16 * 1024, 4, 512, 512),
            (16 * 512, 16, 512, 512 * 4),
            (32 * 1024, 4, 1024, 1024),
            (32 * 1024, 2, 4096, 4096 * 4),
            (64 * 1024, 16, 256, 256 * 4),
        ],
        line_arg="provider",
        line_vals=["einsum", "bmm", "custom"],
        line_names=["einsum", "bmm", "custom"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="latency (ms)",
        plot_name="blockshuffle-performance",
        args={"torch_dtype": torch.float16},
    )
)
def benchmark_blockshuffle(bs, blocks, in_blksz, out_blksz, torch_dtype, provider):
    input = torch.randn(bs, blocks * in_blksz, device="cuda", dtype=torch_dtype) * 0.02
    if in_blksz < out_blksz:
        w1 = (
            torch.randn(blocks, in_blksz, in_blksz, device="cuda", dtype=torch_dtype)
            * 0.02
        )
        w2 = (
            torch.randn(blocks, out_blksz, in_blksz, device="cuda", dtype=torch_dtype)
            * 0.02
        )
    else:
        w1 = (
            torch.randn(blocks, out_blksz, in_blksz, device="cuda", dtype=torch_dtype)
            * 0.02
        )
        w2 = (
            torch.randn(blocks, out_blksz, out_blksz, device="cuda", dtype=torch_dtype)
            * 0.02
        )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "einsum":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: block_shuffle_einsum(input, w1, w2), quantiles=quantiles
        )
    if provider == "bmm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: block_shuffle_bmm(input, w1, w2), quantiles=quantiles
        )
    if provider == "custom":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: block_shuffle_custom(input, w1, w2), quantiles=quantiles
        )
    return ms, max_ms, min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "blocks", "in_blksz", "r_blksz", "out"],
        x_vals=[
            (16 * 1024, 4, 512, 384, 512),
            (16 * 512, 16, 512, 384, 512 * 4),
            (32 * 1024, 4, 1024, 512, 1024),
            (32 * 1024, 2, 1024, 512, 4096 * 4),
            (64 * 1024, 16, 256, 128, 256 * 4),
        ],
        line_arg="provider",
        line_vals=["bmm", "custom"],
        line_names=["bmm", "custom"],
        styles=[("green", "-"), ("green", "--")],
        ylabel="latency (ms)",
        plot_name="block-linear-performance",
        args={"torch_dtype": torch.float16},
    )
)
def benchmark_blockdense(bs, blocks, in_blksz, r_blksz, out, torch_dtype, provider):
    input = torch.randn(bs, in_blksz * blocks, device="cuda", dtype=torch_dtype) * 0.02
    w1 = (
        torch.randn(
            blocks,
            r_blksz,
            in_blksz,
            device="cuda",
            dtype=torch_dtype,
        )
        * 0.02
    )
    w2 = torch.randn(out, r_blksz * blocks, device="cuda", dtype=torch_dtype) * 0.02

    quantiles = [0.5, 0.2, 0.8]
    if provider == "bmm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: block_dense_bmm(input, w1, w2), quantiles=quantiles
        )
    if provider == "custom":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: block_dense_custom(input, w1, w2), quantiles=quantiles
        )
    return ms, max_ms, min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "seq", "d_in", "d_r", "d_out"],
        x_vals=[
            (16, 1024, 4 * 512, 384 * 4, 512 * 4),
            (16, 512, 16 * 512, 384 * 16, 512 * 16),
            (32, 1024, 4 * 1024, 512 * 4, 1024 * 4),
            (32, 1024, 2 * 1024, 512 * 2, 4096 * 2),
            (64, 1024, 16 * 256, 128 * 16, 256 * 16),
        ],
        line_arg="provider",
        line_vals=["custom"],
        line_names=["custom"],
        styles=[("green", "-"), ("green", "--")],
        ylabel="latency (ms)",
        plot_name="block-linear-performance",
        args={"torch_dtype": torch.float16},
    )
)
def benchmark_lowrank(bs, seq, d_in, d_r, d_out, torch_dtype, provider):
    input = torch.randn(bs, seq, d_in, device="cuda", dtype=torch_dtype) * 0.02
    w1 = torch.randn(d_r, d_in, device="cuda", dtype=torch_dtype) * 0.02
    w2 = torch.randn(d_out, d_r, device="cuda", dtype=torch_dtype) * 0.02

    quantiles = [0.5, 0.2, 0.8]
    if provider == "custom":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: low_rank_custom(input, w1, w2), quantiles=quantiles
        )
    return ms, max_ms, min_ms


benchmark_blockshuffle.run(show_plots=True, print_data=True)
benchmark_blockdense.run(show_plots=True, print_data=True)
benchmark_lowrank.run(show_plots=True, print_data=True)
