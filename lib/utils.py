import math
import random
import os, sys
from turtle import width
import imageio
import ipdb

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions import Normal

########### System utils

def seed_torch(seed, verbose=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    if verbose:
        print("==> Set seed to {:}".format(seed))

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)
    
def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print('creating dir: {}'.format(path))
        except OSError:
            print('Failed to create dir: {}'.format(path))
    else:
        print(path, "already exist!")
    return path


########### Calculation utils

def count_parameters(model, requires_grad=False):
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def safe_log(x):
    return torch.log(x.clamp(min=1e-22))

def my_atanh(t):
    return 0.5 * (safe_log(1+t) - safe_log(1-t))

def abnormal(x):
    return ~torch.all(torch.isfinite(x))

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

from torch._six import inf
def grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def geometric_cumulative_reverse_cdf(n_samples, prob):
    cumu_samples = torch.arange(1, n_samples + 1).float()
    return (1. - torch.tensor(prob)) ** (cumu_samples - 1)

class KTailDist(object):
    # theoretically we just need an integer distribution s.t. exist N>N0 for all N0, P(N)>0
    def __init__(self, N=100):
        # p(K >= k) = 1/k
        self.N = N  # position where tail distribution changes.
        self.k = np.arange(1, N + 1)
        self.p = 1 / self.k - 1 / (self.k + 1)
        self.p[-1] = 1 - sum(self.p[:-1])
        self.geom_param = 0.1 # success (stop) prob

    def pN_sample(self):
        sample = np.random.choice(self.k, p=self.p)
        if sample == self.k[-1]:
            # P(N=k) = Geom(k-100;0.1) if k>100
            # if first coin is success, np.random.geometric returns 1
            sample = self.k[-1] + np.random.geometric(self.geom_param) - 1
        return sample
    
    def pN_cumulative_reverse_cdf(self, n_samples):
        # NOTE if the last geometric dist samples a number > 400, will cause index error
        rcdf = 1. / torch.arange(1., self.N)
        rcdf_tail = 1. / self.N * geometric_cumulative_reverse_cdf(
            n_samples - self.N + 1, self.geom_param)
        return torch.cat([rcdf, rcdf_tail])

class MultilevelMasker(object):
    """
    10000000
    11000000
    11110000
    ...
    """
    def __init__(self, init_level=5, base=2):
        self.base = base
        self.level = init_level
        self.init_mask(self.level)
    
    def init_mask(self, level):
        self.mask = torch.zeros(level, self.base ** (level-1)).int()
        for l in range(level):
            self.mask[l, :self.base**l] = 1
    
    def update_mask(self, level):
        if level > self.level:
            new_mask = torch.zeros(level, self.base ** (level-1)).int()
            new_mask[:self.level, :self.base ** (self.level-1)] = self.mask[:, :]
            for l in range(self.level, level):
                new_mask[l, :self.base**l] = 1
            
            self.level = level
            self.mask = new_mask

    def get_mask(self, level):
        if level > self.level:
            self.update_mask(level)
            mask = self.mask
        else:
            mask = self.mask[:level, :self.base ** (level-1)]

        return mask.bool()


class BoundedNormal(object):
    def __init__(self, mean, log_std, std_bound=0.):
        self.mean = mean
        self.log_std = log_std
        self.std_bound = std_bound
    
    def dist(self):
        return Normal(self.mean, self.log_std.exp() + self.std_bound)

    def sample(self, *args):
        return self.dist().sample(*args)

    def rsample(self, *args):
        return self.dist().rsample(*args)

    def log_prob(self, *args):
        return self.dist().log_prob(*args)

    def entropy(self, *args):
        return self.dist().entropy(*args)
    
def cal_mlmc(log_weights, num_level, masker, use_orig=False):
    if use_orig:
        assert masker.base ** (num_level - 1) + 1 == log_weights.shape[-1] 
    else:
        assert masker.base ** (num_level - 1) == log_weights.shape[-1] 
    device = log_weights.device

    if use_orig:
        orig_logw = log_weights[:, 0]
        log_weights = log_weights[:, 1:]
    # assume log_weights.shape = (bs, num_particles)
    log_weights_aug = log_weights.unsqueeze(1).expand(-1, num_level, -1)  ####
    
    # neg_inf = torch.ones_like(log_weights_aug) * -1e20
    neg_inf = log_weights_aug.new_full(log_weights_aug.shape, -np.inf).detach()
    
    mask = masker.get_mask(num_level)[None].to(device)  # (1, num_level, 2 ** num_level)
    # temp = torch.where(mask, log_weights_aug, -np.inf)  
    # temp = torch.where(mask, log_weights_aug.double(), -np.inf).float()
    temp = torch.where(mask, log_weights_aug, neg_inf)
    if use_orig:
        mlcum_iwae = torch.logaddexp(orig_logw[:, None], torch.logsumexp(temp, dim=-1)) \
            - (1 + mask.int().sum(dim=-1)).log().to(device)
        # optional
        mlcum_iwae[:, 0] = torch.logaddexp(orig_logw[:, None], log_weights).mean(-1) - math.log(2)
    else:
        mlcum_iwae = torch.logsumexp(temp, dim=-1) - mask.int().sum(dim=-1).log().to(device)

    mask_shift = torch.zeros_like(mask).float()
    mask_shift[:, 1:, :] = mask.float()[:, :-1, :]
    mask_half = mask.float() - mask_shift
    temp_half = torch.where(mask_half.bool(), log_weights_aug, neg_inf)
    if use_orig:
        mlcum_iwae_half = torch.logaddexp(orig_logw[:, None], torch.logsumexp(temp_half, dim=-1)) \
            - (1 + mask_half.int().sum(dim=-1)).log().to(device)
    else:
        mlcum_iwae_half = torch.logsumexp(temp_half, dim=-1) - mask_half.int().sum(dim=-1).log().to(device)
    
    return mlcum_iwae, mlcum_iwae_half


########### RL utils

from copy import deepcopy
import gym
# import pybullet_envs  # To register tasks in PyBullet
from envs import dmc2gym
import dm_control
from gym.wrappers import TimeLimit


def parse_dmc_envname(name):
    if name == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif name == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    elif "humanoid_CMU" in name:
        # i.e., name="humanoid_CMU_run" -> "humanoid_CMU" + "run"
        domain_name = '_'.join(name.split('_')[:-1])
        task_name = name.split('_')[-1]
    else:
        # i.e., "finger_turn_hard" -> "finger" + "turn_hard"
        domain_name = name.split('_')[0]
        task_name = '_'.join(name.split('_')[1:])
        
    return domain_name, task_name

def make_env(cfg):
    os.environ['MUJOCO_GL']='egl'  # otherwise somehow cannot record video 

    if cfg.dm_control:
        """Helper function to create dm_control environment"""
        domain_name, task_name = parse_dmc_envname(cfg.env)
        print(f"Making dmc environment domain={domain_name}, task={task_name} ...")
        
        if domain_name == 'quadruped':
            camera_id = 2
        elif domain_name == "locom":
            camera_id = 1
        else:
            camera_id = 0
        task_kwargs = dict(random=cfg.seed)
        frame_skip = getattr(cfg, 'action_repeat', 1)
        from_pixels = getattr(cfg, 'from_pixels', False)   
        width = getattr(cfg, "pixels_width", 84) 

        # if domain_name == 'manip':
        #     from dm_control import manipulation
        #     env = manipulation.load(task_name + '_vision')
        # elif domain_name == 'locom':
        #     from dm_control.locomotion.examples import basic_rodent_2020
        #     env = getattr(basic_rodent_2020, task_name)()
        
        if domain_name == "pointmass":  # self-made env
            assert frame_skip == 1
            assert from_pixels == False
            from envs.pointmass_dmc.point_mass_maze import pointmass_gymenv, pointmass_dmcenv
            env = pointmass_gymenv(task_name, cfg.seed, 
                    visualize_reward=True, from_pixels=False)
            # env = pointmass_dmcenv(task_name, task_kwargs=task_kwargs, 
            #             environment_kwargs=None, visualize_reward=False)

        elif domain_name == "quadruped":
            # 不知道为什么一旦import MyDMCWrapper 就 mujoco.FatalError: gladLoadGL error
            env = dmc2gym.make(domain_name=domain_name,
                    task_name=task_name, seed=cfg.seed, frame_skip=frame_skip,
                    visualize_reward=False if from_pixels else True, from_pixels=from_pixels,
                    height=width, width=width, camera_id=camera_id)

        else:  
            from dm_control import suite
            env = suite.load(
                domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs,
                visualize_reward=False, environment_kwargs=None,
            )

            from envs.dmc2gym.my_wrappers import MyDMCWrapper
            env = MyDMCWrapper(env, task_kwargs=task_kwargs, 
                    from_pixels=from_pixels, height=width, width=width, 
                    camera_id=camera_id, frame_skip=frame_skip)
            env = TimeLimit(env, max_episode_steps=(1000 + frame_skip - 1) // frame_skip)

        env.seed(cfg.seed)
        env.action_space.seed(cfg.seed)
        # if hasattr(env.action_space, "np_random"):
        env.action_space.np_random.seed(cfg.seed)

        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1

    else: # openai gym
        print(f"Making gym environment {cfg.env} ...")
        env = gym.make(cfg.env)
        env.seed(cfg.seed)
        env.action_space.seed(cfg.seed)
        env.action_space.np_random.seed(cfg.seed)

    # extract env info
    if env.action_space.__class__.__name__ == "Box":
        # continuous action space
        act_dim = env.action_space.shape[0]
        act_continuous = True
    else:
        assert env.action_space.__class__.__name__ == "Discrete"
        act_dim = env.action_space.n
        act_continuous = False

    # obs_dim = env.observation_space.shape[0]  # include 1-dim done
    obs_dim = np.prod(env.observation_space.shape)
    
    return env, act_dim, act_continuous, obs_dim

def make_eval_env(env, seed):
    eval_env = deepcopy(env)  # deepcopy is necessary
    # eval_env = env
    eval_env.seed(seed)
    eval_env.action_space.seed(seed)
    env.action_space.np_random.seed(seed)
    
    return eval_env

# for dmc suite
def mask_env_state(env_name, remove_velocity=False, remove_position=False):
    domain_name, task_name = parse_dmc_envname(env_name)
    if domain_name == "pointmass":
        from envs.pointmass_dmc.point_mass_maze import pointmass_dmcenv
        env = pointmass_dmcenv(task_name, task_kwargs={"random": 0})
    else:
        env = dm_control.suite.load(domain_name=domain_name, task_name=task_name)
    time_step = env.reset()
    obs = time_step.observation  # OrderedDict
    
    obs_pieces = []
    for k, v in obs.items():
        v_ = np.array([v]) if np.isscalar(v) else v.ravel()
        mask = np.ones_like(v_)
        if (remove_velocity and "velocity" in k) or (remove_position and k == "position"):
            mask = np.zeros_like(mask)
        obs_pieces.append(mask)
    obs_mask = np.concatenate(obs_pieces, axis=0)

    if remove_velocity and ("velocity" not in obs.keys()):
        if "torso_velocity" not in obs.keys():  # for quadruped
            print("WARNING: 'remove_velocity' is set to True, but there's no 'velocity' in observation!")
    if remove_position and ("position" not in obs.keys()):
        print("WARNING: 'remove_position' is set to True, but there's no 'position' in observation!")
    
    return obs_mask.astype(dtype=bool)

# for dm control
class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = os.path.join(root_dir, 'video') if root_dir else None
        if self.save_dir:
            makedirs(self.save_dir)

        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = (self.save_dir is not None) and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array',
                               height=self.height,
                               width=self.width,
                               camera_id=self.camera_id)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            print(f"Video saved at {path}")
