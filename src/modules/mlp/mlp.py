from .basic_mlp import FusedBasicMLP
from ..layer import CustomLinear


class FusedMLP(FusedBasicMLP):
    def __init__(self, hidden_dim, ffn_dim, bias, act="gelu"):
        super().__init__(hidden_dim, ffn_dim, bias, act)
        self.fc1 = CustomLinear(hidden_dim, self.ffn_dim, bias=bias, return_bias=True)
        self.fc2 = CustomLinear(ffn_dim, hidden_dim, bias=bias, return_bias=True)
