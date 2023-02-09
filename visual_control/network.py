from gym import spaces
import math
from typing import Optional, Dict, Tuple
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
# torch.distributions: Independent: turns batch shape into event shape
from torch.distributions import Normal, Independent,\
     Bernoulli, TransformedDistribution, TanhTransform, Categorical

from lib.utils import my_atanh, abnormal
#, count_parameters, MultilevelMasker #, cal_mlmc
from lib.network import EPSILON


class RSSM(nn.Module):
    def __init__(self,
                 action_space: spaces.Box,
                 stoch: int = 30,
                 deter: int = 200,
                 hidden: int = 200,
                 embed: int = 1024,
                 act: nn.Module = nn.ELU, cfg=None):
        super().__init__()
        self.activation = act
        self.stoch_size = stoch
        self.deter_size = deter
        self.hidden_size = hidden
        self.embed_size = embed
        # two "hidden" has different meanings: one is the standard usage, the other means RNN's hidden's size
        self.cell = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.deter_size)
        self.act = act()
        self.rssm_std = 0.1 if cfg is None else cfg.rssm_std 

        # action + state -> GRU input
        rnn_input_size = stoch + action_space.shape[0]
        self.fc_input = nn.Sequential(nn.Linear(rnn_input_size, hidden), self.act)
        # deter state -> next prior
        self.fc_prior = nn.Sequential(
            nn.Linear(deter, hidden), self.act,
            nn.Linear(hidden, 2 * stoch)
        )
        # deter state + image -> next posterior
        self.fc_post = nn.Sequential(
            nn.Linear(deter + embed, hidden), self.act,
            nn.Linear(hidden, 2 * stoch)
        )

    def initial(self, batch_size: int):
        return dict(mean=torch.zeros(batch_size, self.stoch_size, device=self.device),
                    std=torch.zeros(batch_size, self.stoch_size, device=self.device),
                    stoch=torch.zeros(batch_size, self.stoch_size, device=self.device),
                    deter=torch.zeros(batch_size, self.deter_size, device=self.device))
    
    def get_feat(self, state: dict):
        return torch.cat([state['stoch'], state['deter']], -1)

    def get_dist(self, state: dict):
        return Independent(Normal(state['mean'], state['std']), 1)

    def observe(self, embed: Tensor, action: Tensor, state: Optional[Tensor] = None):
        """
        Compute prior and posterior given initial prior, actions and observations.  

        Args:
            embed: (B, T, D) embeded observations
            action: (B, T, D) actions. Note action[t] leads to embed[t]
            state: (B, D) or None, initial state
        Returns:
            post: dict, same key as initial(), each (B, T, D)
            prior: dict, same key as initial(), each (B, T, D)

        Here "state" is params (mean / std / stoch / deter) of a distribution (prior / posterior)
        """
        B, T, _ = action.size()
        if state is None:
            state = self.initial(B)
        post_list = []
        prior_list = []
        for t in range(T):
            post_state, prior_state = self.obs_step(state, action[:, t], embed[:, t])
            prior_list.append(prior_state)
            post_list.append(post_state)
            state = post_state
        prior = {k: torch.stack([state[k] for state in prior_list], dim=1) for k in prior_list[0]}
        post = {k: torch.stack([state[k] for state in post_list], dim=1) for k in post_list[0]}
        return post, prior

    def imagine(self, action: Tensor, state: Optional[Tensor] = None):  # used in video logging
        """
        Compute priors given initial prior and actions.

        Almost the same as observe so nothing special here
        Args:
            action: (B, T, D) actions. Note action[t] leads to embed[t]
            state: (B, D) or None, initial state
        Returns:
            prior: dict, same key as initial(), each (B, T, D)
        """
        B, T, D = action.size()
        if state is None:
            state = self.initial(B)
        assert isinstance(state, dict)
        prior_list = []
        for t in range(T):
            state = self.img_step(state, action[:, t])
            prior_list.append(state)
        prior = {k: torch.stack([state[k] for state in prior_list], dim=1) for k in prior_list[0]}
        return prior

    def obs_step(self, prev_state: Tensor, prev_action: Tensor, embed: Tensor):
        """
        Compute next prior and posterior given previous prior and action
        Args:
            prev_state: (B, D) or None, initial state
            prev_action: (B,  D) actions. 
            embed: (B,  D) embeded observations
        Returns:
            post: dict, same key as initial(), each (B, D)
            prior: dict, same key as initial(), each (B, D)
        """
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self.fc_post(x)
        mean, std = x.chunk(2, dim=-1)
        std = F.softplus(std) + self.rssm_std
        # stoch = self.get_dist(dict(mean=mean, std=std)).rsample()  # slow
        stoch = torch.randn_like(std) * std + mean  # posterior sample
        post = dict(mean=mean, std=std, stoch=stoch, deter=prior['deter'])  # prior['deter'] == post['deter']
        return post, prior

    def img_step(self, prev_state: Tensor, prev_action: Tensor):
        """
        Compute next prior given previous prior and action
        Args:
            prev_state: (B, D) or None, initial state
            prev_action: (B,  D) actions. 
            embed: (B,  D) embeded observations
        Returns:
            post: dict, same key as initial(), each (B, D)
            prior: dict, same key as initial(), each (B, D)
        """
        rnn_input = torch.cat([prev_state['stoch'], prev_action], dim=-1)
        rnn_input = self.fc_input(rnn_input)
        x = deter = self.cell(rnn_input, prev_state['deter'])
        x = self.fc_prior(x)
        mean, std = x.chunk(2, dim=-1)
        std = F.softplus(std) + self.rssm_std
        # stoch = self.get_dist(dict(mean=mean, std=std)).rsample()  # slow
        stoch = torch.randn_like(std) * std + mean  # prior sample
        prior = dict(mean=mean, std=std, stoch=stoch, deter=deter)
        return prior

    @property
    def device(self):
        return next(iter(self.parameters())).device


class ConvEncoder(nn.Module):
    def __init__(self, depth: int = 32, act: nn.Module = nn.ReLU):
        super().__init__()
        self.depth = depth
        self.conv1 = nn.Conv2d(3, 1 * depth, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(1 * depth, 2 * depth, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(2 * depth, 4 * depth, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(4 * depth, 8 * depth, kernel_size=4, stride=2)
        # 64 -> 31 -> 14 -> 6 -> 2
        self.act = act()

    def forward(self, obs: Dict[str, Tensor]):
        # x = obs['image']
        x = obs
        
        T, B, *OTHER = x.size()  # (T, B, 3, 64, 64)
        x = x.view(T * B, *OTHER)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = x.flatten(start_dim=-3)

        TB, *OTHER = x.size()
        x = x.view(T, B, *OTHER)  # (T, B, 1024)

        assert x.size(-1) == 32 * self.depth
        return x


class ConvDecoder(nn.Module):
    def __init__(self, feature_dim: int, depth: int = 32, act: nn.Module = nn.ReLU, shape: Tuple[int, int, int] = (3, 64, 64)):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(feature_dim, 32 * depth)
        self.conv1 = nn.ConvTranspose2d(32 * depth, 4 * depth, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(4 * depth, 2 * depth, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(2 * depth, 1 * depth, kernel_size=6, stride=2)
        self.conv4 = nn.ConvTranspose2d(1 * depth, 3, kernel_size=6, stride=2)
        self.shape = shape
        self.act = act()

    def forward(self, features: torch.Tensor):
        x = self.fc(features)

        T, B, *OTHER = x.size()
        x = x.view(T * B, *OTHER)

        x = x[:, :, None, None]  # (B, C, 1, 1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        mean = self.conv4(x)
        assert mean.size()[-3:] == self.shape
        # Last 3 dims
        TB, *OTHER = mean.size()
        mean = mean.view(T, B, *OTHER)
        return Independent(Normal(mean, 1.0), reinterpreted_batch_ndims=len(self.shape))


class DenseDecoder(nn.Module):
    def __init__(self, input_dim: int, shape: Tuple[int, ...], 
                    layers: int, units: int, dist: str = 'normal', act: nn.Module = nn.ELU):
        super().__init__()
        self.shape = shape
        self.layers = layers
        self.units = units
        self.dist = dist
        self.act = act()
        self.fc_layers = nn.ModuleList()
        for i in range(layers):
            self.fc_layers.append(nn.Linear(input_dim, units))
            input_dim = units
        self.fc_output = nn.Linear(input_dim, int(np.prod(self.shape)))

    def forward(self, features: Tensor):
        x = features
        for layer in self.fc_layers:
            x = layer(x)
            x = self.act(x)
        x = self.fc_output(x)
        x = x.reshape(*x.size()[:-1], *self.shape)

        if self.dist == 'normal':
            return Independent(Normal(x, 1.0), reinterpreted_batch_ndims=len(self.shape))
        elif self.dist == 'binary':
            return Independent(Bernoulli(logits=x), reinterpreted_batch_ndims=len(self.shape))
        else:
            raise ValueError()


class ActionDecoder(nn.Module):
    def __init__(
          self, input_dim, size, layers, units, dist='tanh_normal', act=nn.ELU,
          min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self.size = size
        self.layers = layers
        self.units = units
        self.dist = dist
        self.act = act()
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.fc_layers = nn.ModuleList()
        for i in range(layers):
            self.fc_layers.append(nn.Linear(input_dim, units))
            input_dim = units
        self.fc_output = nn.Linear(units, size * 2 if self.dist == 'tanh_normal' else size)
        self.raw_init_std = math.log(math.exp(init_std - 1))

    def through_network(self, features: Tensor):
        x = features
        for layer in self.fc_layers:
            x = layer(x)
            x = self.act(x)
        return self.fc_output(x)

    def forward(self, features: Tensor):
        x = self.through_network(features)

        if self.dist == 'tanh_normal':
            mean, std = x.chunk(2, dim=-1)
            
            # interesting trick
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)  
            std = F.softplus(std + self.raw_init_std) + self.min_std

            dist = Normal(mean, std)
            dist = TransformedDistribution(dist, TanhTransform())
            dist = Independent(dist, 1)
        
        elif self.dist == 'onehot':
            dist = Categorical(logits=x)
        else:
            raise ValueError()
        return dist

    def sample(self, features: Tensor):
        assert self.dist == 'tanh_normal'
        mean, std = self.through_network(features).chunk(2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        
        # self-written tanh transform
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        log_dtanh = 2. * (math.log(2.) - x_t - F.softplus(-2. * x_t))
        log_prob = normal.log_prob(x_t) - log_dtanh
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def log_prob(self, features, action):
        assert self.dist == 'tanh_normal'
        mean, std = self.through_network(features).chunk(2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std

        action_ = action * (1 - EPSILON)
        atanh_action_ = my_atanh(action_)
        if abnormal(atanh_action_):
            import ipdb; ipdb.set_trace()
        normal = Normal(mean, std)
        log_prob = normal.log_prob(atanh_action_)
        log_prob -= torch.log(1 - action_.pow(2) + EPSILON)  # (num_p, B, T, action_dim)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return log_prob
    
    # def mode(self):
    #     dist: torch.distributions.Distribution = self._dist.expand((self._samples, *self._dist.batch_shape))
    #     sample: torch.Tensor = dist.rsample()
    #     logprob = dist.log_prob(sample)
    #     batch_size = sample.size(1)
    #     feature_size = sample.size(2)
    #     indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
    #     return torch.gather(sample, 0, indices).squeeze(0)


def dreamer_cal_mlmc(log_weights, num_level, masker, use_orig=False):
    # assume log_weights.shape = (num_particles, B, T)
    if use_orig:
        assert masker.base ** (num_level - 1) + 1 == log_weights.shape[0] 
    else:
        assert masker.base ** (num_level - 1) == log_weights.shape[0] 
    device = log_weights.device
    ndim = log_weights.ndim
    assert ndim in [2, 3]

    if use_orig:
        orig_logw = log_weights[0, ...]  # (B, T)
        log_weights = log_weights[1:, ...]  # (num_p, B, T)

    if ndim == 2:
        log_weights_aug = log_weights.unsqueeze(0).expand(num_level, -1, -1)  # (num_level, num_p, BT)
    elif ndim == 3:
        log_weights_aug = log_weights.unsqueeze(0).expand(num_level, -1, -1, -1)  # (num_level, num_p, B, T)
    
    neg_inf = log_weights_aug.new_full(log_weights_aug.shape, -np.inf).detach()
    
    if ndim == 2:
        mask = masker.get_mask(num_level)[:, :, None].to(device)
    elif ndim == 3:
        mask = masker.get_mask(num_level)[:, :, None, None].to(device)  # (num_level, 2 ** num_level, 1, 1)
    temp = torch.where(mask, log_weights_aug, neg_inf)

    if use_orig:
        mlcum_iwae = torch.logaddexp(orig_logw[None], torch.logsumexp(temp, dim=1)) \
            - (1 + mask.int().sum(dim=1)).log().to(device)
        # optional
        mlcum_iwae[0, ...] = torch.logaddexp(orig_logw[None], log_weights).mean(dim=0) - math.log(2)
    else:
        mlcum_iwae = torch.logsumexp(temp, dim=1) - mask.int().sum(dim=1).log().to(device)

    mask_shift = torch.zeros_like(mask).float()
    mask_shift[1:, ...] = mask.float()[:-1, ...]
    mask_half = mask.float() - mask_shift
    temp_half = torch.where(mask_half.bool(), log_weights_aug, neg_inf)
    if use_orig:
        mlcum_iwae_half = torch.logaddexp(orig_logw[None], torch.logsumexp(temp_half, dim=1)) \
            - (1 + mask_half.int().sum(dim=1)).log().to(device)
    else:
        mlcum_iwae_half = torch.logsumexp(temp_half, dim=1) - mask_half.int().sum(dim=1).log().to(device)
    
    return mlcum_iwae, mlcum_iwae_half


if __name__ == '__main__':
    def test_conv():
        encoder = ConvEncoder()
        decoder = ConvDecoder(feature_dim=32 * 8 * 4)
        input = torch.randn(4, 3, 3, 64, 64)
        feature = encoder(dict(image=input))
        output = decoder(feature)

        assert input.size() == output.base_dist.mean.size()

    def test_dense():
        dense_decoder = DenseDecoder(input_dim=64, shape=(7, 13), layers=3, units=128)
        input = torch.randn(4, 64)
        print(dense_decoder(input))

    def test_action():
        action_decoder = ActionDecoder(input_dim=64, size=13, layers=3, units=128)
        input = torch.randn(4, 64)
        print(action_decoder(input).sample().size())

    def test_rssm():
        T = 10
        B = 4
        D = 5
        action = torch.randn(B, T, D)
        embed = torch.randn(B, T, 1024)
        action_space = spaces.Box(low=-1, high=1, shape=(D,))
        rssm = RSSM(action_space)
        prior = rssm.imagine(action)
        prior, post = rssm.observe(embed, action)
        print(prior['deter'].size())
        print(post['deter'].size())

    # test_conv()
    # test_dense()
    # test_action()
    test_rssm()
