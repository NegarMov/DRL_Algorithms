import torch
from torch import nn
from torch import optim


class Critic(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_layers=2, hidden_size=64, lr=1e-2):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        in_size = ob_dim + ac_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())   
            layers.append(nn.Dropout(p=0.1))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.mlp = nn.Sequential(*layers).to(self.device)

        parameters = self.mlp.parameters()

        self.optimizer = optim.Adam(
            parameters,
            lr,
            weight_decay=1e-3
        )

        self.loss_fn = nn.MSELoss()


    def forward(self, ob, ac):
        if not isinstance(ob, torch.Tensor):
            ob = torch.FloatTensor(ob).to(self.device)
        if not isinstance(ac, torch.Tensor):
            ac = torch.FloatTensor(ac).to(self.device)
        
        return self.mlp(torch.cat([ob, ac], dim=-1))
