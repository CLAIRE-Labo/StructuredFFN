from .basic_mlp import FusedBasicMLP
from ..layer import LowRank


class FusedLowRankMLP(FusedBasicMLP):

    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        bias,
        act,
        config,
        init_config,
        device,
    ):
        super().__init__(hidden_dim, ffn_dim, bias, act=act)
        self.fc1 = LowRank(
            hidden_dim,
            self.ffn_dim,
            bias=bias,
            return_bias=True,
            config=config,
            init_config=init_config,
            device=device,
        )
        self.fc2 = LowRank(
            ffn_dim,
            hidden_dim,
            bias=bias,
            return_bias=True,
            config=config,
            init_config=init_config,
            device=device,
        )

    @staticmethod
    def get_ckpt_name(config_method):
        long_name = (
            "r" + str(config_method.rank) + "-" + str(config_method.init.post_init)
        )
        return long_name
