import torch
import torch.nn as nn
from torch import optim


class Critic(nn.Module):
    def __init__(self, ob_dim, hidden_layers=2, hidden_size=64, lr=1e-2):
        super(Critic, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        in_size = ob_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())   
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.mlp = nn.Sequential(*layers).to(self.device)
 
        parameters = self.mlp.parameters()

        self.optimizer = optim.Adam(
            parameters,
            lr
        )

        self.loss_fn = nn.SmoothL1Loss()


    def forward(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)

        return self.mlp(obs)


    def save(self, path):
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)


    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
