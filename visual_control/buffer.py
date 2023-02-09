import torch
from torch.utils.data import Dataset, DataLoader
import random
from gym import spaces
from typing import Union, Optional, List, Dict
import numpy as np


class ReplayBufferDataset(Dataset):

    def __init__(self, replay_buffer, dataset_size, batch_length):
        self.replay_buffer = replay_buffer
        self.dataset_size = dataset_size
        self.batch_length = batch_length

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.replay_buffer.sample_single_episode(self.batch_length)


# efficient buffer (due to the usage of torch dataloader)
class ReplayBuffer: 
    def __init__(self, action_space: spaces.Space, balance: bool = True):
        self.current_episode: Optional[list] = []
        self.action_space = action_space
        self.balance = balance
        self.episodes = []

    def get_iterator(self, train_steps, batch_size, batch_length, num_workers=1, pin_memory=True):
        dataset = ReplayBufferDataset(self, train_steps * batch_size, batch_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, 
            persistent_workers=True)
        return dataloader

    def start_episode(self, obs: dict):
        transition = {"image": obs.copy()}

        transition['action'] = np.zeros(self.action_space.shape)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self.current_episode = [transition]

    def add(self, obs: dict, action: np.ndarray, reward: float, done: bool,
            info: dict):
        transition = {"image": obs.copy()}

        transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount', np.array(1 - float(done)))
        self.current_episode.append(transition)
         
        if done:
            # slow: list of ndarrays -> torch tensor
            # fast: list of ndarrays -> ndarray -> torch tensor
            episode = {
                k: torch.tensor(np.stack([t[k] for t in self.current_episode]), dtype=torch.float32)
                for k in self.current_episode[0]
            }
            self.episodes.append(episode)
            self.current_episode = []

    def sample_single_episode(self, length: int): 
        while True:
            episode = random.choice(self.episodes)
            total = len(next(iter(episode.values())))
            available = total - length
        
            if available < 1:  # actually never happen
                print(f'Skipped short episode of length {available}.')
                continue

            if self.balance: 
                index = min(random.randint(0, total), available)
            else:
                index = int(random.randint(0, available))
            episode = {k: v[index:index + length] for k, v in episode.items()}
            return episode



if __name__ == '__main__':
    from env import make_dmc_env
    import time
    env = make_dmc_env(name='cartpole_swingup')
    replay_buffer = ReplayBuffer(action_space=env.action_space, balance=True)
    steps = 0
    obs = env.reset()
    replay_buffer.start_episode(obs)
    start = time.perf_counter()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action, reward, done, info)
        if done:
            obs = env.reset()
            replay_buffer.start_episode(obs)
        steps += 1
        if steps % 2500 == 0:
            # import ipdb; ipdb.set_trace()
            data = replay_buffer.sample(batch_size=32, length=15)
            for key in data:
                print(key, data[key].shape)
            elapsed = time.perf_counter() - start
            print(
                f'steps: {steps}, frames: {steps * 2}, time: {elapsed:.2f}s, fps: {steps * 2 / elapsed:.2f}'
            )
