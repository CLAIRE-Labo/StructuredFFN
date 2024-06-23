from .basic_mlp import FusedBasicMLP
from ..layer import BlockDense


class FusedBlockDenseMLP(FusedBasicMLP):

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
        self.fc1 = BlockDense(
            hidden_dim,
            self.ffn_dim,
            bias=bias,
            return_bias=True,
            config=config,
            init_config=init_config,
            device=device,
        )
        self.fc2 = BlockDense(
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
            "r"
            + str(config_method.rank)
            + "b"
            + str(config_method.nblocks)
            + "-"
            + str(config_method.init.post_init)
        )
        return long_name
