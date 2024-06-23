import copy
from easydict import EasyDict
from .model import GPT2LMHeadModel
from .mlp import (
    FusedBlockDenseMLP,
    FusedLowRankMLP,
    FusedBlockShuffleMLP,
    FusedMLP,
)

name_to_model = {
    "gpt2": GPT2LMHeadModel,
}

name_to_method = {
    "lowrank": FusedLowRankMLP,
    "blockdense": FusedBlockDenseMLP,
    "blockshuffle": FusedBlockShuffleMLP,
}


def replace_mlp(model, config_method, config_model, device="cuda"):
    first_layer = (
        config_method.kwargs.first_layer
    )  # true: keep the original linear layer
    for i in range(config_model.kwargs.num_layers):
        if first_layer and i == 0:
            continue
        new_module = name_to_method[config_method.name.lower()](
            config_model.kwargs.hidden_dim,
            config_model.kwargs.ffn_dim,
            config_model.kwargs.bias,
            config_model.kwargs.act,
            config_method.kwargs,
            config_model.kwargs.init,
            device=device,
        )
        del model.model.layers[i].mlp
        model.model.layers[i].mlp = new_module


def get_model(config, device="cuda"):
    config_model = config.model
    config_method = config.method
    model = name_to_model[config_model.name.lower()](config_model.get("kwargs", {})).to(
        device
    )

    # replace here
    if config_method.name.lower() == "linear":
        return model
    replace_mlp(model, config_method, config_model, device)
    model.to(device)
    return model


def get_ckpt_name(config):
    config_model = config.model
    config_method = config.method
    long_name = config_model.name + name_to_model[
        config_model.name.lower()
    ].get_ckpt_name(config_model.get("kwargs", {}))
    if config_method.name != "linear":
        long_name += (
            "-"
            + config_method.name
            + name_to_method[config_method.name.lower()].get_ckpt_name(
                config_method.get("kwargs", {})
            )
        )
    return long_name


def update_ratio(module):
    if hasattr(module, "_update_ratio"):
        module._update_ratio()
