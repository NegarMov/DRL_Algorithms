from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque([], maxlen=maxlen)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def append(self, transition):
        self.buffer.append(transition)


    def sample(self, sample_size):
        mini_batch = random.sample(self.buffer, sample_size)

        obs = np.stack([t[0] for t in mini_batch])
        acs = np.stack([t[1] for t in mini_batch])
        next_obs = np.stack([t[2] for t in mini_batch])
        rewards = np.array([t[3] for t in mini_batch])
        terminateds = np.array([t[4] for t in mini_batch])

        return (
            torch.as_tensor(obs, dtype=torch.float32).to(self.device),
            torch.as_tensor(acs, dtype=torch.long).to(self.device),
            torch.as_tensor(next_obs, dtype=torch.float32).to(self.device),
            torch.as_tensor(rewards, dtype=torch.float32).to(self.device),
            torch.as_tensor(terminateds, dtype=torch.bool).to(self.device)
        )


    def __len__(self):
        return len(self.buffer)
