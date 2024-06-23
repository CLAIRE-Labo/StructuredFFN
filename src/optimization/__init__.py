from torch import optim
from .scheduler import *


name_to_scheduler = {
    "multisteplr": lambda optimizer, kwargs: _MultiStepLR(optimizer, **kwargs),
    "cosineannealinglr": lambda optimizer, kwargs: _CosineAnnealingLR(
        optimizer, **kwargs
    ),
}

name_to_optimizer = {
    "adam": lambda params, kwargs: optim.Adam(params, **kwargs),
    "sgd": lambda params, kwargs: optim.SGD(params, **kwargs),
    "adamw": lambda params, kwargs: optim.AdamW(params, **kwargs),
}


def get_lr_scheduler(config_optimization, optimizer):
    name = config_optimization.lr_scheduler.name.lower()
    return name_to_scheduler[name](optimizer, config_optimization.lr_scheduler.kwargs)


def get_optimizer(config_optimization, params):
    name = config_optimization.optimizer.name.lower()
    return name_to_optimizer[name](params, config_optimization.optimizer.kwargs)
