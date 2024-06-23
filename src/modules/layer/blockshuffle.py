import torch.nn as nn
import torch
from .basiclinear import BasicLinear
from ..op import block_shuffle_bmm, block_shuffle_custom


class BlockShuffle(BasicLinear):

    def __init__(
        self,
        in_features,
        out_features,
        bias,
        return_bias,
        config,
        init_config,
        device="cuda",
    ):
        super().__init__(
            in_features, out_features, bias, return_bias, config, init_config, device
        )
        self.nblocks = config["nblocks"]
        assert self.in_features % self.nblocks == 0
        assert self.out_features % self.nblocks == 0

        in_blksz = self.in_features // self.nblocks
        out_blksz = self.out_features // self.nblocks

        if self.in_features < self.out_features:
            self.blkdiag1 = nn.Parameter(
                torch.empty(self.nblocks, in_blksz, in_blksz, device=device)
            )
            self.blkdiag2 = nn.Parameter(
                torch.empty(self.nblocks, out_blksz, in_blksz, device=device)
            )
        else:
            self.blkdiag1 = nn.Parameter(
                torch.empty(self.nblocks, out_blksz, in_blksz, device=device)
            )
            self.blkdiag2 = nn.Parameter(
                torch.empty(self.nblocks, out_blksz, out_blksz, device=device)
            )
        self._init_weights()
        self.post_init()

    def get_weights(
        self,
    ):
        return [self.blkdiag1, self.blkdiag2]

    @torch.no_grad()
    def post_init(
        self,
    ):
        if self.config.init.post_init == "ortho":
            for i in range(self.nblocks):
                U, S, Vh = torch.linalg.svd(self.blkdiag1.data[i], full_matrices=False)
                self.blkdiag1.data[i] = torch.mm(U, Vh)
                U, S, Vh = torch.linalg.svd(self.blkdiag2.data[i], full_matrices=False)
                self.blkdiag2.data[i] = torch.mm(U, Vh)

        # init guide linear
        if hasattr(self, "guide_linear"):
            self.guide_linear.data = torch.mm(
                torch.block_diag(*torch.unbind(self.blkdiag2.data, dim=0)),
                torch.block_diag(*torch.unbind(self.blkdiag1.data, dim=0)),
            )

    def forward(self, input):
        out = block_shuffle_custom(input, self.blkdiag1, self.blkdiag2)
        return self.forward_guide_layer(input, out)

    def extra_repr(self) -> str:
        return f"blockdiag1={self.blkdiag1.shape}, blockdiag2={self.blkdiag2.shape}, bias={self.bias is not None}, guide={self.training_config.enabled}"
