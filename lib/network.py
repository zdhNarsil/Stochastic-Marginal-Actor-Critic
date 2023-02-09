from os import stat
import ipdb
import math
from cmath import isfinite

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from lib.utils import BoundedNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6  # 1e-10 is not enough for the atanh issue


def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    net = nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))
    return net

def weights_init_(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.orthogonal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers=1, std_bound=0.):
        super(GaussianNetwork, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) \
                                                for _ in range(num_hidden_layers)])
        self.mean_linear = nn.Linear(hidden_dim, output_dim)
        self.log_std_linear = nn.Linear(hidden_dim, output_dim)

        self.apply(weights_init_)
        
        self.std_bound = std_bound

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=-1)
        x = F.relu(self.linear1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, *inputs, num_particles=None):
        mean, log_std = self.forward(*inputs)
        normal = Normal(mean, log_std.exp() + self.std_bound)

        kld = - log_std.mul(2).add(1) + log_std.exp().pow(2) + mean.pow(2)
        kld = kld.mul_(0.5).sum(dim=-1, keepdim=True)  # (bs, 1) or (T, bs, 1)

        if num_particles:
            latent = normal.rsample((num_particles,))  # (num_p, bs, output_dim) or (num_p, T, bs, dim)
            log_prob = normal.log_prob(latent).sum(-1) # (num_p, bs) or (num_p, T, bs)

            kld = kld.repeat_interleave(num_particles, dim=-1)  # (bs, num_p) or (T, bs, num_p)
        else:
            latent = normal.rsample()
            log_prob = normal.log_prob(latent).sum(-1, keepdim=True)  # (bs,)

        return latent, log_prob, kld

    def log_prob(self, x, *inputs):
        mean, log_std = self.forward(*inputs)
        std = log_std.exp() + self.std_bound
        normal = Normal(mean, std)
        return normal.log_prob(x)

    def get_dist(self, *inputs):
        mean, log_std = self.forward(*inputs)
        return BoundedNormal(mean, log_std, std_bound=self.std_bound)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_hidden_layers=1, 
                        act=nn.ReLU, init_weights=True):
        super(QNetwork, self).__init__()

        # Q1 architecture: 2 + num_hidden_layers in total
        self.Q1 = nn.Sequential(nn.Linear(num_inputs, hidden_dim), act(),
                                *sum([[nn.Linear(hidden_dim, hidden_dim), act()]
                                                        for _ in range(num_hidden_layers)]
                                    , []),
                                nn.Linear(hidden_dim, 1))
        # Q2 architecture
        self.Q2 = nn.Sequential(nn.Linear(num_inputs, hidden_dim), act(),
                                *sum([[nn.Linear(hidden_dim, hidden_dim), act()]
                                                        for _ in range(num_hidden_layers)]
                                    , []),
                                nn.Linear(hidden_dim, 1))

        self.apply(weights_init_)

    def forward(self, *inputs):
        xu = torch.cat(inputs, dim=-1)
        
        x1 = self.Q1(xu)
        x2 = self.Q2(xu)

        return x1, x2