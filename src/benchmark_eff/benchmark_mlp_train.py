import argparse
import numpy as np
import torch.nn as nn
import torch
import random
import os
import hydra
import triton
import time
import sys
from hide_warnings import hide_warnings

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

print(project_root)
sys.path.insert(0, project_root)

from src.modules.mlp import (
    FusedBlockDenseMLP,
    FusedLowRankMLP,
    FusedBlockShuffleMLP,
    FusedMLP,
)

# pure bfloat16 efficiency

name_to_method = {
    "lowrank": FusedLowRankMLP,
    "blockdense": FusedBlockDenseMLP,
    "blockshuffle": FusedBlockShuffleMLP,
    "linear": FusedMLP,
}
from omegaconf import DictConfig, OmegaConf

torch_dtype = torch.bfloat16


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def benchmark_train(net, inp):
    def fn(input):
        net(input)

    quantiles = [0.5, 0.2, 0.8]

    t, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(inp), quantiles=quantiles, warmup=50, rep=200
    )
    latency = t
    throughput = inp.shape[0] * inp.shape[1] / latency * 10**3
    print("Latency (ms): {}, Throughput (token/s): {}".format(latency, throughput))
    return latency, throughput


@hide_warnings(out=False)
@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="refinedweb_config",
)
def main(config):
    OmegaConf.register_new_resolver("eval", eval)
    config_model = config.model
    config_method = config.method
    f = open("../../exp/logs/fig3.log", "a+")

    if config_method.name == "linear":
        model = (
            FusedMLP(
                config_model.kwargs.hidden_dim,
                config_model.kwargs.ffn_dim,
                config_model.kwargs.bias,
                config_model.kwargs.act,
            )
            .cuda()
            .to(torch_dtype)
        )
    else:
        model = (
            name_to_method[config.method.name.lower()](
                config_model.kwargs.hidden_dim,
                config_model.kwargs.ffn_dim,
                config_model.kwargs.bias,
                config_model.kwargs.act,
                config_method.kwargs,
                config_model.kwargs.init,
                device="cuda",
            )
            .cuda()
            .to(torch_dtype)
        )
    model.eval()
    with torch.no_grad():
        input = (
            torch.randn(
                config.data.test.test_batch, 1024, config_model.kwargs.hidden_dim
            )
            .cuda()
            .to(torch_dtype)
        )
        latency, throughput = benchmark_train(model, input)
    if config_method.name == "linear":
        print(
            f"{config_model.kwargs.hidden_dim}, {config_model.kwargs.ffn_dim}", file=f
        )
    else:
        print(
            f"{config_model.kwargs.hidden_dim}, {config_model.kwargs.ffn_dim}, {model.get_ckpt_name(config_method.kwargs)}",
            file=f,
        )
    print(
        f"latency: {latency}, throughput: {throughput}, bs: {config.data.test.test_batch}, params: {sum(p.numel() for p in model.parameters())}",
        file=f,
    )


if __name__ == "__main__":
    set_seed(1005)
    main()
    print("******END*******")
