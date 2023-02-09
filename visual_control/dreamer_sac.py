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
from lib.utils import soft_update, hard_update
from lib.network import QNetwork

act_dict = {
    'relu': nn.ReLU,
    'elu': nn.ELU
}
class DreamerSAC(nn.Module):
    def __init__(self, cfg, action_space: spaces.Box):
        super().__init__()
        self.action_space = action_space
        self.actdim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        self.cfg = cfg
        self.metrics = collections.defaultdict(AverageMeter)
        
        ########## world model
        cnn_act = act_dict[cfg.cnn_act]
        act = act_dict[cfg.dense_act]
        self.encoder = ConvEncoder(depth=cfg.cnn_depth, act=cnn_act)
        self.dynamics = RSSM(self.action_space, stoch=cfg.stoch_size,
                             deter=cfg.deter_size, hidden=cfg.deter_size, cfg=cfg)

        feat_size = cfg.stoch_size + cfg.deter_size
        self.decoder = ConvDecoder(feature_dim=feat_size, depth=cfg.cnn_depth, act=cnn_act)
        self.reward = DenseDecoder(input_dim=feat_size, shape=(), layers=2, units=cfg.num_units, act=act)
        if self.cfg.pcont:
            self.pcont = DenseDecoder(input_dim=feat_size, shape=(), layers=3, units=cfg.num_units, dist='binary', act=act)
        
        ########## policy optimization
        # input size of actor and critic network
        feat_size = cfg.stoch_size + cfg.deter_size
        self.actor = ActionDecoder(input_dim=feat_size, size=self.actdim, layers=4,
                                   units=cfg.num_units, dist=cfg.action_dist, 
                                   init_std=cfg.action_init_std, act=act)
        self.critic = QNetwork(feat_size + self.actdim, cfg.num_units, num_hidden_layers=2, act=act, init_weights=False)
        self.critic_target = QNetwork(feat_size + self.actdim, cfg.num_units, num_hidden_layers=2, act=act, init_weights=False)
        hard_update(self.critic_target, self.critic)
        
        self.model_modules = nn.ModuleList([self.encoder, self.decoder, self.dynamics, self.reward])
        if self.cfg.pcont:
            self.model_modules.append(self.pcont)

        self.model_optimizer = optim.Adam(self.model_modules.parameters(), lr=cfg.model_lr,
                                          weight_decay=cfg.weight_decay)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.lr, 
                                          weight_decay=cfg.weight_decay)

        self.alpha = cfg.alpha
        if self.cfg.auto_tune:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape)).item()
            self.target_entropy += math.log(cfg.nmode)
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self.log_alpha.data.fill_(math.log(cfg.init_alpha))  # Initialize alpha to be high.
            self.alpha_optim = optim.Adam([self.log_alpha], lr=cfg.lr)

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
        
        # model loss
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

        feature = torch.cat([post['stoch'], post['deter'].detach()], -1)
        feature = feature.detach() # necessary
        # Actor loss
        with freeze(nn.ModuleList([self.critic])):
            action, log_pi = self.actor.sample(feature)
            qf1_pi, qf2_pi = self.critic(feature.detach(), action)  
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        # critic loss
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(feature[:, 1:])  # "next_state"
            qf1_next_target, qf2_next_target = self.critic_target(feature[:, 1:], next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            next_q_value = data['reward'][:, 1:, None] + self.cfg.discount * min_qf_next_target 

        qf1, qf2 = self.critic(feature[:, :-1].detach(), data['action'][:, 1:])  # feat[t - 1] + action[t] -> feat[t]
        critic_loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)

        self.model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)

        (model_loss + critic_loss + actor_loss).backward()
        
        model_norm = nn.utils.clip_grad_norm_(self.model_modules.parameters(), self.cfg.grad_clip)
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.model_optimizer.step()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        if self.cfg.auto_tune:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        soft_update(self.critic_target, self.critic, self.cfg.tau)

        self.scalar_summaries(
            data, feature, prior_dist, post_dist, likes, div,
            model_loss, critic_loss, actor_loss, alpha_loss, 
            model_norm, critic_norm, actor_norm, log_pi, qf1)
        
        if log_video:
            self.image_summaries(data, embed, image_pred, video_path)

    @torch.no_grad()
    def scalar_summaries(
          self, data, feat, prior_dist, post_dist, likes, div,
          model_loss, critic_loss, actor_loss, alpha_loss, 
          model_norm, critic_norm, actor_norm, log_pi, q_func):
        self.metrics['model_grad_norm'].update_state(model_norm)
        self.metrics['critic_grad_norm'].update_state(critic_norm)
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
        self.metrics['critic_loss'].update_state(critic_loss)
        self.metrics['actor_loss'].update_state(actor_loss)
        self.metrics['alpha_loss'].update_state(alpha_loss)
        self.metrics['alpha'].update_state(self.alpha)
        self.metrics['log_pi'].update_state(log_pi.mean())
        self.metrics['q_func'].update_state(q_func.mean())

    @torch.no_grad()
    def image_summaries(self, data, embed, image_pred, video_path):
        # Take the first 6 sequences in the batch
        B, T, C, H, W = image_pred.mean.size()
        B = 6
        truth = data['image'][:6] + 0.5
        recon = image_pred.mean[:6]
        init, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])  # get posterior
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['action'][:6, 5:], init)
        feat = self.dynamics.get_feat(prior)

        openl = self.decoder(feat).mean
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], dim=1)
        error = (model - truth + 1) / 2
        # (B, T, 3, 3H, W)
        openl = torch.cat([truth, model, error], dim=3)
        # (T, 3H, B * W, 3)
        openl = openl.permute(1, 3, 0, 4, 2).reshape(T, 3 * H, B * W, C).cpu().numpy()
        openl = (openl * 255.).astype(np.uint8)
        imageio.mimsave(video_path, openl, fps=30)
        print(f"Video saved at {video_path}")

    def preprocess_batch(self, data: Dict[str, np.ndarray]):
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
       # If no state yet initialise tensors otherwise take input state
        if state is None:
            latent = self.dynamics.initial(len(obs))
            action = torch.zeros((len(obs), self.actdim), dtype=torch.float32).to(self.cfg.device)
        else:
            latent, action = state
        embed = self.encoder(self.preprocess_observation(obs))
        embed = embed.squeeze(0)
        latent, _ = self.dynamics.obs_step(latent, action, embed)  
        feat = torch.cat([latent['stoch'], latent['deter']], -1)

        # If training sample random actions if not pick most likely action 
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
        self.critic.train()
        self.actor.train()
    
    def eval(self):
        self.model_modules.eval()
        self.critic.eval()
        self.actor.eval()
