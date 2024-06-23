import math
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingLR,
)


__all__ = [
    "_MultiStepLR",
    "_CosineAnnealingLR",
]


class _MultiStepLR(MultiStepLR):

    def __init__(self, optimizer, **kwargs):
        kwargs["milestones"] = [
            int(e * kwargs.pop("T_max")) for e in kwargs["milestones"]
        ]
        super(_MultiStepLR, self).__init__(optimizer, **kwargs)


class _CosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, **kwargs):
        self.warmup_iter = 0
        if "warmup_iter" in kwargs:
            self.warmup_iter = int(kwargs.pop("warmup_iter") * kwargs["T_max"])
        super(_CosineAnnealingLR, self).__init__(optimizer, **kwargs)

    def get_lr(self):
        if self.last_epoch < self.warmup_iter:
            return [
                (self.last_epoch + 1) / self.warmup_iter * base_lr
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_iter)
                    / (self.T_max - self.warmup_iter)
                )
            )
            / 2
            for base_lr in self.base_lrs
        ]
