import imageio
import collections
import torch
import json
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
from torch import Tensor, nn, optim
from torch.nn import functional as F
from functools import partial
import math
from termcolor import colored
from torch.distributions import kl_divergence

from visual_control.utils import Timer, AttrDict, freeze, AverageMeter
from visual_control.network import ConvDecoder, ConvEncoder, ActionDecoder, DenseDecoder, RSSM

act_dict = {
    'relu': nn.ReLU,
    'elu': nn.ELU
}
class Dreamer(nn.Module):
    def __init__(self, cfg, action_space: spaces.Box):
        super().__init__()
        self.action_space = action_space
        self.actdim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        self.cfg = cfg
        self.metrics = collections.defaultdict(AverageMeter)
        
        self.build_model()

    def build_model(self):
        cnn_act = act_dict[self.cfg.cnn_act]
        act = act_dict[self.cfg.dense_act]
        self.encoder = ConvEncoder(depth=self.cfg.cnn_depth, act=cnn_act)
        self.dynamics = RSSM(self.action_space, stoch=self.cfg.stoch_size,
                             deter=self.cfg.deter_size, hidden=self.cfg.deter_size, cfg=self.cfg)

        feat_size = self.cfg.stoch_size + self.cfg.deter_size
        self.decoder = ConvDecoder(feature_dim=feat_size, depth=self.cfg.cnn_depth, act=cnn_act)
        self.reward = DenseDecoder(input_dim=feat_size, shape=(), layers=2, units=self.cfg.num_units, act=act)
        if self.cfg.pcont:
            self.pcont = DenseDecoder(input_dim=feat_size, shape=(), layers=3, units=self.cfg.num_units, dist='binary', act=act)
        self.value = DenseDecoder(input_dim=feat_size, shape=(), layers=3, units=self.cfg.num_units, act=act)
        self.actor = ActionDecoder(input_dim=feat_size, size=self.actdim, layers=4,
                                   units=self.cfg.num_units, dist=self.cfg.action_dist, 
                                   init_std=self.cfg.action_init_std, act=act)
        
        self.model_modules = nn.ModuleList([self.encoder, self.decoder, self.dynamics, self.reward])
        if self.cfg.pcont:
            self.model_modules.append(self.pcont)

        self.model_optimizer = optim.Adam(self.model_modules.parameters(), lr=self.cfg.model_lr,
                                          weight_decay=self.cfg.weight_decay)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.cfg.value_lr,
                                          weight_decay=self.cfg.weight_decay)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr,
                                          weight_decay=self.cfg.weight_decay)

        if self.cfg.tf_init:
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight.data)
                    if hasattr(m.bias, 'data'):
                        m.bias.data.fill_(0.0)

    def update(self, data: Dict[str, Tensor], log_video: bool, video_path: Path=None):
        """
        Corresponds to Dreamer._train.
        Update the model and policy/value. Log metrics and video.
        """
        data = self.preprocess_batch(data)
        
        with torch.autocast(device_type="cuda"):  
            embed = self.encoder(data['image'])  # (B, T, 1024)
            post, prior = self.dynamics.observe(embed, data['action'])
            
            feat = self.dynamics.get_feat(post)  # (B, T, 230)
            image_pred = self.decoder(feat)  # dist on (B, T, 3, H, W), std=1.0
            reward_pred = self.reward(feat)  # dist on (B, T)

        likes = AttrDict()
        # mean over batch and time, sum over pixel
        likes.image = image_pred.log_prob(data['image']).mean(dim=[0, 1])
        likes.reward = reward_pred.log_prob(data['reward']).mean(dim=[0, 1])
        if self.cfg.pcont:
            pcont_pred = self.pcont(feat)
            pcont_target = self.cfg.discount * data['discount']
            likes.pcont = torch.mean(pcont_pred.log_prob(pcont_target), dim=[0, 1])
            likes.pcont *= self.cfg.pcont_scale

        prior_dist = self.dynamics.get_dist(prior)
        post_dist = self.dynamics.get_dist(post)
        div = kl_divergence(post_dist, prior_dist).mean(dim=[0, 1])
        div = torch.clamp(div, min=self.cfg.free_nats)  # in case of prior = posterior => kl = 0
        model_loss = self.cfg.kl_scale * div - sum(likes.values())

        # Actor loss
        with freeze(nn.ModuleList([self.model_modules, self.value])):
            # (H + 1, BT, D), indexed t = 0 to H, includes the 
            # start state unlike original implementation
            imag_feat, log_pi = self.imagine_ahead(post)  # (H+1, B*T, 230)
            if self.cfg.ts: # Thompson Sampling
                reward = self.reward(imag_feat[1:]).rsample()
            else:
                reward = self.reward(imag_feat[1:]).mean  # (H, B*T)
            if self.cfg.pcont:
                if self.cfg.ts:
                    pcont = self.pcont(imag_feat[1:]).rsample()
                else:
                    pcont = self.pcont(imag_feat[1:]).mean
            else:
                pcont = self.cfg.discount * torch.ones_like(reward)  # (H, B*T)

            if self.cfg.ts:
                value = self.value(imag_feat[1:]).rsample()
            else:
                value = self.value(imag_feat[1:]).mean
            with torch.no_grad():
                # (H, BT, D) discount[t] -> state[t] is terminal or after a terminal state
                discount = torch.cat([torch.ones_like(pcont[:1]), torch.cumprod(pcont, dim=0)[:-1]], dim=0)

            if not self.cfg.single_step_q:
                returns = torch.zeros_like(value)  # (H, B*T)
                last = value[-1]
                for t in reversed(range(self.cfg.horizon)):
                    returns[t] = (reward[t] + pcont[t] * (
                        (1. - self.cfg.disclam) * value[t] + self.cfg.disclam * last))
                    last = returns[t]
                actor_loss = -(discount * returns).mean(dim=[0, 1])
            else:
                q_estimates = torch.zeros_like(value)
                for t in range(self.cfg.horizon):
                    q_estimates[t] = reward[t] + pcont[t] * value[t]
                actor_loss = -(discount * q_estimates).mean(dim=[0, 1])

        # Value loss
        target = returns.detach()
        if self.cfg.update_horizon is None:
            value_pred = self.value(imag_feat[:-1].detach())
            value_loss = torch.mean(-value_pred.log_prob(target) * discount, dim=[0, 1])
        else:
            value_pred = self.value(imag_feat[:self.cfg.update_horizon].detach())
            value_loss = torch.mean(-value_pred.log_prob(target[:self.cfg.update_horizon]) * discount[:self.cfg.update_horizon], dim=[0, 1])

        self.model_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)

        (value_loss + model_loss + actor_loss).backward()

        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        value_norm = nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_clip)
        model_norm = nn.utils.clip_grad_norm_(self.model_modules.parameters(), self.cfg.grad_clip)
        self.actor_optimizer.step()
        self.model_optimizer.step()
        self.value_optimizer.step()

        self.scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, 
            model_norm, value_norm, actor_norm, log_pi,
            value_pred.mean)
        if log_video:
            self.image_summaries(data, embed, image_pred, video_path)

    @torch.no_grad()
    def scalar_summaries(
          self, data, feat, prior_dist, post_dist, likes, div,
          model_loss, value_loss, actor_loss, model_norm, value_norm,
          actor_norm, log_pi, value):
        self.metrics['model_grad_norm'].update_state(model_norm)
        self.metrics['value_grad_norm'].update_state(value_norm)
        self.metrics['actor_grad_norm'].update_state(actor_norm)
        self.metrics['prior_ent'].update_state(prior_dist.entropy().mean())
        self.metrics['post_ent'].update_state(post_dist.entropy().mean())
        self.metrics['action_ent'].update_state(self.actor(feat).\
                        base_dist.base_dist.entropy().sum(dim=-1).mean())
        self.metrics['action_logstd'].update_state(
            self.actor(feat).base_dist.base_dist.scale.log().mean())  # avg across (B, T, act_dim)
        
        for name, logprob in likes.items():
            self.metrics[name + '_loss'].update_state(-logprob)
        self.metrics['div'].update_state(div)
        self.metrics['model_loss'].update_state(model_loss)
        self.metrics['value_loss'].update_state(value_loss)
        self.metrics['actor_loss'].update_state(actor_loss)
        self.metrics['value_func'].update_state(value.mean())
        self.metrics['log_pi'].update_state(log_pi.mean())

    @torch.no_grad()
    def image_summaries(self, data, embed, image_pred, video_path):
        # Take the first 6 sequences in the batch
        B, T, C, H, W = image_pred.mean.size()  # T=50
        B = 6
        truth = data['image'][:6] + 0.5
        recon = image_pred.mean[:6]
        init, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])  # get posterior
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['action'][:6, 5:], init)

        openl = self.decoder(self.dynamics.get_feat(prior)).mean
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], dim=1)
        error = (model - truth + 1) / 2
        # (B, T, 3, 3H, W)
        openl = torch.cat([truth, model, error], dim=3)
        # (T, 3H, B * W, 3)
        openl = openl.permute(1, 3, 0, 4, 2).reshape(T, 3 * H, B * W, C).cpu().numpy()
        openl = (openl * 255.).astype(np.uint8)
        imageio.mimsave(video_path, openl, fps=30)

    def preprocess_batch(self, data: Dict[str, Tensor]):
        data = {k: torch.as_tensor(v, device=self.cfg.device, dtype=torch.float) for k, v in data.items()}
        data['image'] = data['image'] / 255.0 - 0.5
        clip_rewards = dict(none=lambda x: x, tanh=torch.tanh)[self.cfg.clip_rewards]
        data['reward'] = clip_rewards(data['reward'])
        return data

    def preprocess_observation(self, obs: Dict[str, np.ndarray]):
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float)
        obs = obs / 255.0 - 0.5
        return obs

    @torch.no_grad()
    def get_action(self, obs: Dict[str, np.ndarray], state: Optional[Tensor] = None, training: bool = True) \
            -> Tuple[np.ndarray, Optional[Tensor]]:
        """
        Corresponds to Dreamer.__call__, but without training.
        Args:
            obs: obs['image'] shape (C, H, W), uint8
            state: None, or Tensor
        Returns:
            action: (D)
            state: None, or Tensor
        """
        # Add T and B dimension for a single action
        obs = obs[None, None, ...]

        action, state = self.policy(obs, state, training)  #self.action_space.sample(),None
        action = action.squeeze(axis=0)
        return action, state

    def policy(self, obs: Tensor, state: Tensor, training: bool) -> Tensor:
        """
        Args:
            obs: (B, C, H, W)
            state: (B, D)
        Returns:
            action: (B, D)
            state: (B, D)
        """
        if state is None:
            latent = self.dynamics.initial(len(obs))
            action = torch.zeros((len(obs), self.actdim), dtype=torch.float32).to(self.cfg.device)
        else:
            latent, action = state
        embed = self.encoder(self.preprocess_observation(obs))
        embed = embed.squeeze(0)
        latent, _ = self.dynamics.obs_step(latent, action, embed)  # get posterior
        
        feat = self.dynamics.get_feat(latent)
        if training:
            action = self.actor(feat).sample()
        else:
            # this is dirty: it should be the mode
            # the original repo samples 100 times and takes the argmax of log_prob
            action = torch.tanh(self.actor(feat).base_dist.base_dist.mean)  # base_dist is gaussian

        action = self.exploration(action, training)
        state = (latent, action)
        action = action.cpu().detach().numpy()
        action = np.array(action, dtype=np.float32)
        return action, state

    def exploration(self, action: Tensor, training: bool) -> Tensor:
        """
        Args:
            action: (B, D)
        Returns:
            action: (B, D)
        """
        if training:
            amount = self.cfg.expl_amount
            if self.cfg.expl_min:
                amount = max(self.cfg.expl_min, amount)
            self.metrics['expl_amount'].update_state(amount)
        elif self.cfg.eval_noise:
            amount = self.cfg.eval_noise
        else:
            return action

        if self.cfg.expl == 'additive_gaussian':
            return torch.clamp(torch.normal(action, amount), 
                            self.action_space.low.min(), self.action_space.high.max())
        if self.cfg.expl == 'completely_random':
            return torch.rand(action.shape, -1, 1)
        if self.cfg.expl == 'epsilon_greedy':
            indices = torch.distributions.Categorical(0 * action).sample()
            return torch.where(
                        torch.rand(action.shape[:1], 0, 1) < amount,
                        torch.one_hot(indices, action.shape[-1], dtype=self.float),
                        action)
        raise NotImplementedError(self.cfg.expl)

    def imagine_ahead(self, post: dict) -> Tensor:  
        """
        Starting from a posterior, do rollout using your currenct policy.

        Args:
            post: dictionary of posterior state. Each (B, T, D)
        Returns:
            imag_feat: (T, B, D). concatenation of imagined posteiror states. 
        """
        if self.cfg.pcont:
            # (B, T, D)
            # last state may be terminal. Terminal's next discount prediction is not trained.
            post = {k: v[:, :-1] for k, v in post.items()}
        # (B, T, D) -> (BT, D)
        flatten = lambda x: x.reshape(-1, *x.size()[2:])  # (B, T, ...) -> (B*T, ...)
        start = {k: flatten(v).detach() for k, v in post.items()}
        state = start
        
        state_list = [start]
        log_pi_ls = []
        for i in range(self.cfg.horizon):
            if self.cfg.update_horizon is not None and i >= self.cfg.update_horizon:
                with torch.no_grad():  # truncate gradient
                    action = self.actor(self.dynamics.get_feat(state).detach()).rsample()
            else:
                # This is what the original implementation does: state is detached
                action, log_pi_i = self.actor.sample(self.dynamics.get_feat(state).detach())
                log_pi_ls.append(log_pi_i)
            
            with torch.autocast(device_type="cuda"): 
                state = self.dynamics.img_step(state, action)
            state_list.append(state)
            if self.cfg.single_step_q:
                # Necessary, if you are using single step q estimate
                state = {k: v.detach() for k, v in state.items()}

        # (H, BT, D)
        states = {k: torch.stack([state[k] for state in state_list], dim=0) for k in state_list[0]}
        imag_feat = self.dynamics.get_feat(states)
        log_pi = torch.stack(log_pi_ls, dim=0).squeeze()  # (H, BT, 1) -> (H, BT)
        return imag_feat, log_pi

    def load(self, path: Union[str, Path], device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        path = Path(path)
        with path.open('wb') as f:
            self.load_state_dict(torch.load(f, map_location=device))

    # Change to state dict if we just want to save the weights
    def save(self, path: Union[str, Path]): 
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('wb') as f:
            torch.save(self.state_dict(), f)

    def train(self):
        self.model_modules.train()
        self.value.train()
        self.actor.train()
    
    def eval(self):
        self.model_modules.eval()
        self.value.eval()
        self.actor.eval()
