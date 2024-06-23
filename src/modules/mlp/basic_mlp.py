import torch.nn as nn
from transformer_engine.pytorch.jit import set_jit_fusion_options
from ..op import bias_gelu_impl, bias_swiglu_impl


act_func_dict = {
    "gelu": bias_gelu_impl,
    "swiglu": bias_swiglu_impl,
}


class FusedBasicMLP(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, bias, act="gelu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        # fuse bias and gelu
        set_jit_fusion_options()
        self.fc1 = None
        self.fc2 = None
        self.act_func = act_func_dict.get(act, None)
        if act in ["swiglu"]:
            self.ffn_dim *= 2

    def forward(self, input):
        fc1_outs = self.fc1(input)
        gelu_out = self.act_func(*fc1_outs)
        fc2_outs = self.fc2(gelu_out)
        return fc2_outs
