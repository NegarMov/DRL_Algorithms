from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, sample_size):
        mini_batch = random.sample(self.buffer, sample_size)

        obs = np.stack([t[0] for t in mini_batch])
        acs = np.stack([t[1] for t in mini_batch])
        next_obs = np.stack([t[2] for t in mini_batch])
        rewards = np.array([t[3] for t in mini_batch])
        terminateds = np.array([t[4] for t in mini_batch], dtype=bool)

        return (
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(acs, dtype=torch.long),
            torch.as_tensor(next_obs, dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(terminateds)
        )


    def __len__(self):
        return len(self.buffer)
