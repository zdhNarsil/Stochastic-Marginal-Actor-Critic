import time
import datetime
from contextlib import contextmanager
from torch import nn, Tensor
import torch
import numpy as np


class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


class AverageMeter:
    def __init__(self):
        self.reset_states()

    def reset_states(self):
        self.total = 0.0
        self.count = 0

    def update_state(self, scalar):
        if isinstance(scalar, Tensor):
            scalar = scalar.item()
        self.total += scalar
        self.count += 1

    def result(self):
        return self.total / self.count

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.perf_counter()
        self.last_time = time.perf_counter()

    def split(self):
        elapsed_time = time.perf_counter() - self.last_time
        self.last_time = time.perf_counter()
        return elapsed_time

    def total(self) -> float:
        return time.perf_counter() - self.start_time

    def total_human(self) -> str:
        return str(datetime.timedelta(seconds=self.total()))


@contextmanager
def freeze(module: nn.Module):
    """
    Disable gradient for all module parameters. However, if input requires grad
    the graph will still be constructed.
    """
    try:
        prev_states = [p.requires_grad for p in module.parameters()]
        for p in module.parameters():
            p.requires_grad_(False)
        yield
        
    finally:
        for p, state in zip(module.parameters(), prev_states):
            p.requires_grad_(state)
