import torch.nn as nn
import torch
from ..op import block_dense_custom
from .basiclinear import BasicLinear


class BlockDense(BasicLinear):

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
        self.rank = config["rank"]
        self.nblocks = config["nblocks"]
        assert self.in_features % self.nblocks == 0
        assert self.rank % self.nblocks == 0
        self.blkdiag = nn.Parameter(
            torch.empty(
                self.nblocks,
                self.rank // self.nblocks,
                self.in_features // self.nblocks,
                device=device,
            )
        )
        self.lr = nn.Parameter(torch.empty(self.out_features, self.rank, device=device))

        self._init_weights()
        self.post_init()

    def get_weights(
        self,
    ):
        return [self.blkdiag, self.lr]

    @torch.no_grad()
    def post_init(
        self,
    ):
        if self.config.init.post_init == "ortho":
            for i in range(self.nblocks):
                U, S, Vh = torch.linalg.svd(self.blkdiag.data[i], full_matrices=False)
                self.blkdiag.data[i] = torch.mm(U, Vh)
            U, S, Vh = torch.linalg.svd(self.lr.data, full_matrices=False)
            self.lr.data = torch.mm(U, Vh)
        # init guide linear
        if hasattr(self, "guide_linear"):
            self.guide_linear.data = torch.mm(
                self.lr.data, torch.block_diag(*torch.unbind(self.blkdiag.data, dim=0))
            )

    def forward(self, input):
        out = block_dense_custom(input, self.blkdiag, self.lr)
        return self.forward_guide_layer(input, out)

    def extra_repr(self) -> str:
        return f"blockdiag1={self.blkdiag.shape}, linear={self.lr.shape}, bias={self.bias is not None}, guide={self.training_config.enabled}"
