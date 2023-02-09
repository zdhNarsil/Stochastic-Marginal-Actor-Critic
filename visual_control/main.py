from dataclasses import dataclass
import datetime
import json
from omegaconf import OmegaConf, open_dict
from typing import Optional, Tuple, List, Dict, Union
import os, sys
os.environ['MUJOCO_GL']='egl'
import wandb
import logging
from termcolor import colored
from pathlib import Path
import pickle as pkl
import hydra

import numpy as np
import torch

from envs.pomdp_dmc.wrappers import POMDPWrapper  # DMC, take env as input
from visual_control.buffer import ReplayBuffer
from visual_control.utils import Timer
from lib.utils import seed_torch, make_env, make_eval_env, mask_env_state, VideoRecorder, makedirs

"""
python -m visual_control.main dreamer=0 
python -m visual_control.main dreamer=1
python -m visual_control.main dreamer=2 estimate=nmlmc qagg=lse 
"""

log = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

@dataclass
class Config:
    # General.
    log_every: int = int(1e3)
    
    # Environment.
    eval_noise: float = 0.0
    clip_rewards: str = 'none'
    
    # Model.
    dense_act: str = 'elu'
    cnn_act: str = 'relu'
    cnn_depth: int = 32
    free_nats: float = 3.0
    kl_scale: float = 1.0
    pcont: bool = False  # whether predict done (predicting done is meaningless for DMC)
    pcont_scale: float = 10.0  # not used when pcont is False
    weight_decay: float = 0.0

    # Training.
    train_every: int = 1000
    model_lr: float = 6e-4
    value_lr: float = 8e-5
    actor_lr: float = 8e-5
    grad_clip: float = 100.0
    dataset_balance: bool = False

    # Behavior.
    discount: float = 0.99
    disclam: float = 0.95
    horizon: int = 15  # imaginary rollout length
    action_dist: str = 'tanh_normal'
    action_init_std: float = 5.0
    expl: str = 'additive_gaussian'
    expl_amount: float = 0.3
    expl_decay: float = 0.0
    expl_min: float = 0.0

    # Ablations.
    update_horizon: Optional[int] = None  # policy value after this horizon are not updated
    single_step_q: bool = False  # Use 1-step target as an estimate of q.
    tf_init: bool = False


def get_env_name(cfg):
    env_name = f"{cfg.env}" \
        + (f"_flick{cfg.flicker}" if cfg.flicker > 0 else "") \
        + (f"_noise{cfg.noise}" if cfg.noise > 0 else "") \
        + (f"_miss{cfg.missing}" if cfg.missing > 0 else "")
    return env_name

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)
        log.info(f"workspace: {self.work_dir}")
        print(f"workspace: {self.work_dir}")
        
        self.global_steps = 0
        self.i_episode = 0
        self.device = torch.device(f"cuda:{self.cfg.device:d}" if torch.cuda.is_available() else "cpu")
        
        train_env, self.act_dim, _, self.obs_dim = make_env(cfg)
        with open_dict(self.cfg):
            self.cfg.action_range = [float(train_env.action_space.low.min()), 
                                     float(train_env.action_space.high.max())]
            self.cfg.auto_tune = self.cfg.alpha < 0  # negative means use entropy auto tuning
            if self.cfg.deter_size in [50, 100]:
                self.cfg.stoch_size = 100 
        
        self.save_dir = self.work_dir
        if cfg.dreamer == 0:
            from visual_control.dreamer import Agent
        elif cfg.dreamer == 1:
            from visual_control.dreamer_sac import DreamerSAC as Agent
        elif cfg.dreamer == 2:
            from visual_control.smac import SMAC as Agent
        else:
            raise NotImplementedError

        self.agent = Agent(cfg, train_env.action_space).to(cfg.device)
        self.replay_buffer = ReplayBuffer(action_space=train_env.action_space, balance=False)
        self.records = {"reward_train": {}, "reward_test": {}} 

        self.video_dir = os.path.join(self.work_dir, 'video')
        if self.cfg.video:
            self.video_recorder = VideoRecorder(self.work_dir if cfg.video else None)
            makedirs(self.video_dir)

    def run(self):
        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)

        log.info(f"Work directory is {self.work_dir}")
        print(f"Work directory is {self.work_dir}")
        log.info("Running with configuration:\n" + str(self.cfg))  # OmegaConf.to_yaml(self.cfg, resolve=True)

        train_env, self.act_dim, _, self.obs_dim = make_env(self.cfg)
        eval_env = make_eval_env(train_env, self.cfg.seed + 1)
        train_env = POMDPWrapper(train_env, flicker_prob=self.cfg.flicker, noise_sigma=self.cfg.noise, sensor_missing_prob=self.cfg.missing)
        eval_env = POMDPWrapper(eval_env, flicker_prob=self.cfg.flicker, noise_sigma=self.cfg.noise, sensor_missing_prob=self.cfg.missing)
        log.info(f"obs_dim={self.obs_dim} act_dim={self.act_dim} self.max_trajectory_len={train_env._max_episode_steps}")

        job_type = "sweep" if ("/checkpoint/" in self.work_dir) else "local"
        with open_dict(self.cfg):
            self.cfg.job_type = job_type
        cfg = self.cfg

        wandb_name = get_env_name(cfg)
        use_pomdp = False if wandb_name == f"{cfg.env}" else True
        wandb_name += ("" if cfg.seed == 0 else f"_seed{cfg.seed}")

        # for model learning part
        wandb_name += (f"_deter{cfg.deter_size}" if cfg.deter_size != 200 else "")
        wandb_name += (f"_stoch{cfg.stoch_size}" if cfg.stoch_size != 30 else "")
        wandb_name += (f"_nunit{cfg.num_units}" if cfg.num_units != 300 else "")
        wandb_name += (f"_rssmstd{cfg.rssm_std:.1e}" if cfg.rssm_std != 0.1 else "")
        
        # for policy opt part
        if cfg.dreamer == 1:
            wandb_name += "_sac"
        if cfg.dreamer == 2:
            wandb_name += f"_smac"

        if cfg.dreamer >= 1:
            wandb_name += ("_autot" if cfg.auto_tune else f"_alpha{cfg.alpha}")
            wandb_name += (f"_saclr{cfg.lr:.1e}" if cfg.lr != 3e-4 else "")
            if cfg.dreamer in [2,]:
                wandb_name += (f"_pazent{cfg.pazent}" if cfg.pazent > 0. else "")
                wandb_name += (f"_aggfirst" if cfg.aggfirst else "")
                wandb_name += f"_qagg{cfg.qagg}"
                wandb_name += f"_{cfg.estimate}"
                if cfg.estimate in ["naive", "nmlmc",]:
                    wandb_name += f"{cfg.num_p}"
        
        print(f"Wandb name: {wandb_name}")
        if cfg.wandb:
            wandb.init(project="Dreamer_POMDP" if use_pomdp else "Dreamer", entity="generative-modeling", group=self.cfg.env,
                 name=wandb_name, dir=self.work_dir, resume=False, config=cfg, save_code=True, job_type=job_type)
            log.info(f"Wandb initialized for {wandb_name}")
        
        seed_torch(self.cfg.seed)

        # Main Loop
        obs = train_env.reset()
        self.replay_buffer.start_episode(obs)
        agent_state = None
        self.timer = Timer()
        self.last_frames = 0
        
        episode_reward = 0
        episode_steps = 0

        self.evaluate(eval_env)  # Initial evaluation
        self.save()
        while self.global_frames < self.cfg.num_steps:
            if self.global_frames < self.cfg.start_steps:
                action = train_env.action_space.sample()
            else:
                action, agent_state = self.agent.get_action(obs, agent_state, training=True)
            obs, reward, done, info = train_env.step(action)
            self.replay_buffer.add(obs, action, reward, done, info)
            self.global_steps += 1
            episode_reward += reward
            episode_steps += 1
            
            if done:
                self.i_episode += 1
                self.log(episode_reward, episode_steps, prefix='Train')
                if self.cfg.wandb:
                    wandb.log({"reward_train": episode_reward}, step=self.global_frames)
                self.records["reward_train"][self.i_episode] = episode_reward
                # Reset
                episode_reward = 0
                episode_steps = 0
                obs = train_env.reset()
                self.replay_buffer.start_episode(obs)
                agent_state = None

            # Training
            if self.global_frames >= self.cfg.start_steps and self.global_frames % self.cfg.train_every == 0:
                self.agent.train()

                dataloader = self.replay_buffer.get_iterator(self.cfg.train_steps, self.cfg.batch_size, self.cfg.batch_length)
                for train_step, data in enumerate(dataloader):
                    log_video = self.cfg.video and \
                            self.global_frames % self.cfg.eval_every == 0 and train_step == 0
                    self.agent.update(data, log_video=log_video,
                            video_path=os.path.join(self.video_dir, f'error_{self.global_frames_str}.gif'))

                if self.global_frames % self.cfg.log_every == 0:
                    if self.cfg.wandb:
                        wandb_dict = {
                            "loss/image": self.agent.metrics['image_loss'].result(),
                            "loss/reward": self.agent.metrics['reward_loss'].result(),
                            "loss/model": self.agent.metrics['model_loss'].result(),
                            "loss/actor": self.agent.metrics['actor_loss'].result(),
                            
                            "ent/prior": self.agent.metrics['prior_ent'].result(),
                            "ent/posterior": self.agent.metrics['post_ent'].result(),
                            "KL_div": self.agent.metrics['div'].result(),
                        }
                        if cfg.dreamer in [0,]:
                            wandb_dict.update({
                                "loss/value": self.agent.metrics['value_loss'].result(),
                                "ent/action": self.agent.metrics['action_ent'].result(),
                                "ent/paz_logstd": self.agent.metrics["action_logstd"].result(),
                                "value_func": self.agent.metrics['value_func'].result(),
                                "log_pi": self.agent.metrics['log_pi'].result(),
                            })

                        else:
                            if cfg.dreamer == 1:
                                wandb_dict.update({
                                    "ent/action": self.agent.metrics['action_ent'].result()
                                    })
                            wandb_dict.update({
                                "loss/critic": self.agent.metrics['critic_loss'].result(),
                                "loss/alpha": self.agent.metrics['alpha_loss'].result(),
                                "ent/paz_logstd": self.agent.metrics["action_logstd"].result(),
                                "alpha": self.agent.metrics['alpha'].result(),
                                "log_pi": self.agent.metrics['log_pi'].result(),
                                "q_func": self.agent.metrics['q_func'].result(),
                            })

                        if cfg.dreamer in [2,]:
                            wandb_dict.update({"ent/feature_std": self.agent.metrics["feature_std"].result()})
                        
                        wandb.log(wandb_dict, step=self.global_frames)

                    metrics = [(k, float(v.result())) for k, v in self.agent.metrics.items()]
                    [m.reset_states() for m in self.agent.metrics.values()]
                    print(colored(f'[{self.global_frames}]', 'yellow'), ' / '.join(f'{k} {v:.1f}' for k, v in metrics))

            if self.global_frames % self.cfg.eval_every == 0:
                avg_reward = self.evaluate(eval_env)
                if self.cfg.wandb:
                    wandb.log({"reward_test": avg_reward}, step=self.global_frames)
                    if self.cfg.video:
                        wandb.log({
                            f"{self.global_frames_str}":
                                            wandb.Video(os.path.join(self.video_recorder.save_dir,
                                                    f'eval_{self.global_frames_str}.gif')),
                            f"error_{self.global_frames}":
                                            wandb.Video(os.path.join(self.video_dir,
                                                    f'error_{self.global_frames_str}.gif'))
                            })
                self.save()

        train_env.close()
        eval_env.close()
        if self.cfg.wandb:
            wandb.finish()

    def evaluate(self, eval_env):
        lengths_ls = []
        episode_reward_ls = []
        self.agent.eval()
        for epi_idx in range(self.cfg.num_eval_episodes):
            if self.cfg.video:
                self.video_recorder.init(enabled= (epi_idx == 0))

            obs = eval_env.reset()
            episode_reward = 0
            length = 0
            agent_state = None
            done = False
            while not done:
                action, agent_state = self.agent.get_action(obs, agent_state, training=False)
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward
                length += 1

                if self.cfg.video:
                    self.video_recorder.record(eval_env)

            episode_reward_ls.append(episode_reward)
            lengths_ls.append(length)

            if self.cfg.video:
                self.video_recorder.save(f'eval_{self.global_frames_str}.gif')

        avg_reward = float(np.mean(episode_reward_ls))
        avg_length = float(np.mean(lengths_ls))
        self.log(avg_reward, avg_length, prefix='Test')
        self.records["reward_test"][self.i_episode] = avg_reward
        return avg_reward

    def save(self, tag="latest"):
        for dir in {self.save_dir, self.work_dir}:
            self.agent.save(os.path.join(dir, f"{tag}.ckpt"))

            path = os.path.join(dir, f"records_{tag}.json")
            with open(path, "wb") as f:
                pkl.dump(self.records, f)
            print(f"Saved at {path}")

    def log(self, avg_return: float, avg_length: float, prefix: str):
        colored_prefix = colored(prefix, 'yellow' if prefix == 'Train' else 'green')
        elapsed_time = self.timer.split()
        total_time = datetime.timedelta(seconds=int(self.timer.total()))
        fps = (self.global_frames - self.last_frames) / elapsed_time
        self.last_frames = self.global_frames
        print(f'{colored_prefix:<14} | Frame: {self.global_frames} | Episode: {self.i_episode} | '
              f'Reward: {avg_return:.2f} | Length: {avg_length:.1f} | FPS: {fps:.2f} | Time: {total_time}')

    @property
    def global_frames(self):
        return self.global_steps * self.cfg.action_repeat

    @property
    def global_frames_str(self):
        length = len(str(self.cfg.num_steps))
        return f'frame{self.global_frames:0{length}d}'


from visual_control.main import Workspace as W
from visual_control.main import Config as C
# @hydra.main(version_base="1.1", config_path="configs", config_name="main")
@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    config = OmegaConf.structured(C)
    cfg = OmegaConf.merge(cfg, config)
    
    fname = os.path.join(os.getcwd(), "latest.pkl")
    if os.path.exists(fname):
        log.info(f"Resuming fom {fname}")
        with open(fname, "rb") as f: 
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    try:
        workspace.run()
    except Exception as e:
        log.critical(e, exc_info=True)

if __name__ == '__main__':
    main()