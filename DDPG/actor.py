import torch
from torch import nn
from torch import optim


class Actor(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_layers=2, hidden_size=64, lr=1e-2):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        in_size = ob_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())   
            in_size = hidden_size
        layers.append(nn.Linear(in_size, ac_dim))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers).to(self.device)

        parameters = self.mlp.parameters()

        self.optimizer = optim.Adam(
            parameters,
            lr
        )


    def forward(self, ob):
        if not isinstance(ob, torch.Tensor):
            ob = torch.FloatTensor(ob).to(self.device)

        return self.mlp(ob)


    def save(self, path):
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)


    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
