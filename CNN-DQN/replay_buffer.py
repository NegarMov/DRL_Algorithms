import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, maxlen, obs_shape, obs_dtype=np.uint8, device=None):
        self.maxlen = maxlen
        self.obs_buf = np.empty((maxlen, *obs_shape), dtype=obs_dtype)
        self.next_obs_buf = np.empty((maxlen, *obs_shape), dtype=obs_dtype)
        self.acts_buf = np.empty((maxlen,), dtype=np.int64)
        self.rews_buf = np.empty((maxlen,), dtype=np.float32)
        self.dones_buf = np.empty((maxlen,), dtype=bool)
        self.ptr = 0
        self.size = 0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def append(self, transition):
        ob, ac, next_ob, rew, done = transition
        self.obs_buf[self.ptr] = ob
        self.acts_buf[self.ptr] = ac
        self.next_obs_buf[self.ptr] = next_ob
        self.rews_buf[self.ptr] = rew
        self.dones_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.from_numpy(self.obs_buf[idxs]).float().to(self.device)
        acs = torch.from_numpy(self.acts_buf[idxs]).long().to(self.device)
        next_obs = torch.from_numpy(self.next_obs_buf[idxs]).float().to(self.device)
        rews = torch.from_numpy(self.rews_buf[idxs]).float().to(self.device)
        dones = torch.from_numpy(self.dones_buf[idxs]).bool().to(self.device)
        return obs, acs, next_obs, rews, dones

    def __len__(self):
        return self.size
