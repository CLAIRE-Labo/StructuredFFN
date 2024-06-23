import torch.nn as nn
import torch

from .util import LinearTempDecay, CosineTempDecay


class BasicLinear(nn.Module):

    def __init__(
        self, in_features, out_features, bias, return_bias, config, init_config, device
    ):
        super().__init__()
        # config: method part, and model init
        self.device = device
        self.config = config
        self.init_config = init_config
        self.training_config = self.config.training
        # model part
        self.in_features = in_features
        self.out_features = out_features
        # otherwise, we need to fuse the bias into the ops
        assert return_bias is True
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, device=device))
        else:
            self.bias = None

        if self.training_config.enabled:
            self.guide_linear = nn.Parameter(
                torch.empty(self.out_features, self.in_features, device=device)
            )
            self.register_buffer("count", torch.tensor(0).cuda(), persistent=True)
            self.register_buffer("ratio", torch.tensor(1.0).cuda(), persistent=True)
            guide_scheduler = {
                "linear": LinearTempDecay,
                "cosine": CosineTempDecay,
            }
            self.guide_scheduler = guide_scheduler[self.training_config.scheduler](
                t_max=self.training_config.max_step
            )

    @torch.no_grad()
    def _update_ratio(
        self,
    ):
        self.count += 1
        self.ratio = self.guide_scheduler(self.count)

    def _check_guide_layer(
        self,
    ):
        if not self.training_config.enabled:
            return False
        if (
            self.training_config.reduce_flop
            and torch.rand_like(self.ratio) >= self.ratio
        ):
            return False
        return True

    def forward_guide_layer(self, input, out):
        if self._check_guide_layer():
            guide_out = torch.matmul(input, self.guide_linear.transpose(-1, -2))
            out = self.ratio * guide_out + (1.0 - self.ratio) * out
        return out, self.bias

    def get_weights(
        self,
    ):
        pass

    @torch.no_grad()
    def _init_weights(
        self,
    ):
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        for para in self.get_weights():
            if self.init_config.weight_init == "xavier":
                nn.init.normal_(para, mean=0.0, std=(para.shape[-1] ** -0.5))
            elif self.init_config.weight_init == "fixed":
                nn.init.normal_(para, std=self.init_config.initializer_range)
            else:
                raise NotImplementedError
