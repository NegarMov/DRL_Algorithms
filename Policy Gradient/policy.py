import itertools
import torch
import torch.nn as nn
from torch import optim
from torch import distributions


class Policy(nn.Module):
    def __init__(self, ob_dim, ac_dim, discrete, hidden_layers=2, hidden_size=64, lr=1e-2):
        super(Policy, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.discrete = discrete

        layers = []
        in_size = ob_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())   
            in_size = hidden_size
        layers.append(nn.Linear(in_size, ac_dim))
        self.mlp = nn.Sequential(*layers).to(self.device)
 
        if discrete:
            parameters = self.mlp.parameters()
        else:
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=self.device)
            )
            parameters = itertools.chain([self.logstd], self.mlp.parameters())

        self.optimizer = optim.Adam(
            parameters,
            lr
        )


    def forward(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)

        if self.discrete:
            logits = self.mlp(obs)
            probs = nn.functional.softmax(logits, dim=0)
            return distributions.Categorical(probs)
        else:
            mean = self.mlp(obs)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)


    def save(self, path):
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)


    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
