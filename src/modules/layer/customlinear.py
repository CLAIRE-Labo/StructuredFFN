import torch.nn as nn
import torch


# please do not inhere the basic linear here
class CustomLinear(nn.Module):

    def __init__(self, in_features, out_features, bias, return_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
            )
        )
        # otherwise, we need to fuse the bias into the ops
        assert return_bias is True

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, inp):
        output = torch.matmul(inp, self.weight.transpose(-1, -2))
        return output, self.bias

    def extra_repr(self) -> str:
        return f"linearshape={self.weight.shape}, bias={self.bias is not None}"
