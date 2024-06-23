import torch.nn as nn
import torch
from .basiclinear import BasicLinear
from ..op import low_rank_custom


class LowRank(BasicLinear):

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
            in_features,
            out_features,
            bias,
            return_bias,
            config,
            init_config,
            device=device,
        )
        self.rank = config["rank"]
        self.lr1 = nn.Parameter(torch.empty(self.rank, self.in_features, device=device))
        self.lr2 = nn.Parameter(
            torch.empty(self.out_features, self.rank, device=device)
        )
        self._init_weights()
        self.post_init()

    def get_weights(
        self,
    ):
        return [self.lr1, self.lr2]

    @torch.no_grad()
    def post_init(
        self,
    ):
        if self.config.init.post_init == "svd":
            org_linear = nn.Parameter(
                torch.empty(self.out_features, self.in_features, device=self.device)
            )
            if self.init_config.weight_init == "xavier":
                nn.init.normal_(
                    org_linear, mean=0.0, std=(org_linear.shape[-1] ** -0.5)
                )
            elif self.init_config.weight_init == "fixed":
                nn.init.normal_(org_linear, std=self.init_config.initializer_range)
            else:
                raise NotImplementedError
            U, S, Vh = torch.linalg.svd(org_linear, full_matrices=False)
            sqrt_S = torch.sqrt(torch.diag_embed(S[: self.rank]))
            self.lr1.data = sqrt_S @ Vh[: self.rank, :]
            self.lr2.data = U[:, : self.rank] @ sqrt_S

        # init guide linear
        if hasattr(self, "guide_linear"):
            self.guide_linear.data = torch.mm(self.lr2.data, self.lr1.data)

    def forward(self, input):
        out = low_rank_custom(input, self.lr1, self.lr2)
        return self.forward_guide_layer(input, out)

    def extra_repr(self) -> str:
        return f"lr1={self.lr1.shape}, lr2={self.lr2.shape}, bias={self.bias is not None}, guide={self.training_config.enabled}"
