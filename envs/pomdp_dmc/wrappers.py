import ipdb
import numpy as np
import gym
from gym import spaces


class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env, mask=None,
        flicker_prob=0., noise_sigma=0., sensor_missing_prob=0.):

        assert isinstance(env, gym.Env)
        super().__init__(env)
        self._max_episode_steps = self.env._max_episode_steps
        self.flicker_prob = flicker_prob
        self.noise_sigma = noise_sigma
        self.sensor_missing_prob = sensor_missing_prob

        # self.remain_obs_idx = np.arange(0, self.env.reset().squeeze().shape[0])
        self.mask = mask
    
    def observation(self, obs):
        # if self.remove_velocity:
            # obs = obs.flatten()[self.remain_obs_idx]
        
        if obs.ndim == 3:  # pixel-based DMC is in uint8, (3, 64, 64) 
            assert self.mask is None
            assert obs.dtype == "uint8"

            if self.flicker_prob > 0.:
                obs = np.zeros_like(obs)
            if self.sensor_missing_prob > 0.:
                obs[np.random.rand(*obs.shape)  <= self.sensor_missing_prob] = 0
            if self.noise_sigma > 0.:
                obs = obs / 255.0 - 0.5
                noise = np.random.normal(0, self.noise_sigma, obs.shape) #.astype(np.float32)
                obs = np.clip(obs + noise, -0.5, 0.5) + 0.5
                obs = (obs * 255.).astype(np.uint8)
            
            return obs

        elif obs.ndim == 1:
            if self.mask is not None:
                obs = obs[self.mask]

            if self.flicker_prob > 0.:
                if np.random.rand() <= self.flicker_prob:
                    obs = np.zeros_like(obs).flatten()
                else:
                    obs = obs.flatten()
            
            if self.noise_sigma > 0.:
                # somehow the gaussian noise is double (float64)
                noise = np.random.normal(0, self.noise_sigma, obs.shape).astype(np.float32)
                obs = (obs + noise).flatten()
            
            if self.sensor_missing_prob > 0.:
                obs[np.random.rand(len(obs)) <= self.sensor_missing_prob] = 0
                obs = obs.flatten()

            return obs.astype(np.float32)  # numpy uses double by default
        
        else:
            raise NotImplementedError