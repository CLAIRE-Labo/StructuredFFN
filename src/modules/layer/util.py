import torch
import math


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0, start_b=1.0, end_b=0.0):
        self.t_max = t_max
        self.warmup = warm_up
        self.start_b = torch.tensor(start_b).cuda()
        self.end_b = torch.tensor(end_b).cuda()
        print(
            "linear scheduler for self-guided training in steps {} with warmup {}".format(
                self.t_max, self.warmup
            )
        )

    def __call__(self, t):
        if t < self.warmup:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.warmup) / (self.t_max - self.warmup)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class CosineTempDecay:
    def __init__(self, t_max=20000, warm_up=0, start_b=1.0, end_b=0.0):
        self.t_max = t_max
        self.warmup = warm_up
        self.start_b = torch.tensor(start_b).cuda()
        self.end_b = torch.tensor(end_b).cuda()
        print(
            "Cosine scheduler for self-guided training in steps {} with warmup {}".format(
                self.t_max, self.warmup
            )
        )

    def __call__(self, t):
        if t < self.warmup:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.warmup) / (self.t_max - self.warmup)
            return self.end_b + 0.5 * (self.start_b - self.end_b) * (
                1 + torch.cos(rel_t * math.pi)
            )
