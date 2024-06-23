import argparse
import numpy as np
import torch.nn as nn
import torch
import random
import os
import hydra
import time
import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

print(project_root)
sys.path.insert(0, project_root)
from hide_warnings import hide_warnings
from src.modules import get_model
from omegaconf import DictConfig, OmegaConf

torch_dtype = torch.bfloat16
prefill = 0


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


@torch.no_grad()
def benchmark_infer(net, generation, inp=None):
    net.eval()
    net.to(torch_dtype)
    seq_len = inp.shape[1]
    bs = inp.shape[0]
    inference_params = net.prepare_inference_params(
        bs,
        mx_seq=seq_len + generation,
    )
    tokens = bs * (seq_len + generation)
    repeat = 10
    warmup = 10
    for i in range(warmup):
        if seq_len > 0:
            inference_params.sequence_len_offset = 0
            net(inp, inference_params=inference_params)
        cur = torch.zeros(bs, 1).long().cuda()
        for j in range(seq_len, seq_len + generation):
            inference_params.sequence_len_offset = j
            net(input_ids=cur, inference_params=inference_params, use_cache=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(repeat):
        if seq_len > 0:
            inference_params.sequence_len_offset = 0
            net(inp, inference_params=inference_params)
        cur = torch.zeros(bs, 1).long().cuda()
        for j in range(seq_len, seq_len + generation):
            inference_params.sequence_len_offset = j
            net(input_ids=cur, inference_params=inference_params, use_cache=True)
    torch.cuda.synchronize()
    t1 = time.time()
    latency = (t1 - t0) / repeat * (10**3)
    throughput = tokens * repeat / (t1 - t0)
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
    model = get_model(config)
    model.eval()
    f = open("../../exp/logs/arch_infer_latency.log", "a+")
    print(
        "h-f-a-nq-nkv",
        config.model.kwargs.hidden_dim,
        config.model.kwargs.ffn_dim,
        config.model.kwargs.attn_dim,
        config.model.kwargs.num_q_heads,
        config.model.kwargs.num_kv_heads,
        file=f,
    )
    if config.method.name != "linear":
        print(config.method.kwargs, file=f)
    input_ids = torch.zeros(config.data.test.test_batch, prefill).long().cuda()
    latency, throughput = benchmark_infer(
        model, config.model.kwargs.max_position_embeddings, input_ids
    )
    params = sum(p.numel() for p in model.parameters())
    params_woemb = params - 32000 * config.model.kwargs.hidden_dim

    print(
        f"bs: {config.data.test.test_batch}, generation: {config.model.kwargs.max_position_embeddings}, latency: {latency}, params: {params}, params_woemb: {params_woemb}, throughput: {throughput}",
        file=f,
    )


if __name__ == "__main__":
    set_seed(1005)
    main()
    print("******END*******")
