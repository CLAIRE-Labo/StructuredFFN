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

from src.modules import get_model
from omegaconf import DictConfig, OmegaConf
from hide_warnings import hide_warnings

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


def func(batch_size, generation, model):
    try:
        torch.cuda.empty_cache()
        cur = torch.zeros(batch_size, 1).long().cuda()  # test the last token directly

        inference_params = model.prepare_inference_params(
            batch_size,
            mx_seq=prefill + generation,
        )
        inference_params.sequence_len_offset = prefill + generation - 1
        model(input_ids=cur, inference_params=inference_params, use_cache=True)
    except RuntimeError as e:
        return None
    return batch_size


@torch.no_grad()
def find_max_batch_size(model, generation):
    start = 256
    batch_size = start
    max_batch_size = start
    step = 256
    while True:
        if func(batch_size, generation, model):
            max_batch_size = batch_size
            batch_size += step
        else:
            break
    print(f"bs: {max_batch_size}")
    return max_batch_size


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
    model.to(torch_dtype)
    f = open("../../exp/logs/arch_bs.log", "a+")
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
    bs = find_max_batch_size(model, config.model.kwargs.max_position_embeddings)
    print(
        f"bs: {bs}, generation: {config.model.kwargs.max_position_embeddings}",
        file=f,
    )


if __name__ == "__main__":
    set_seed(1005)
    main()
    print("******END*******")
