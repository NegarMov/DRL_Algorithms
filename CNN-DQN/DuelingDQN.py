import math

import torch
from torch import nn
from torch import optim

class DuelingDQN(nn.Module):
    def __init__(self, ac_dim, lr=1e-2):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=7*7*64, out_features=512),
            nn.ReLU()
        )
        self.V = nn.Linear(512, 1).to(self.device)
        self.A = nn.Linear(512, ac_dim).to(self.device)

        self.initialize_weights()

        self.to(self.device)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr
        )

        self.loss_fn = nn.SmoothL1Loss()


    def initialize_weights(self):
        std = math.sqrt(2.0 / (4 * 84 * 84))
        nn.init.normal_(self.conv_block[0].weight, mean=0.0, std=std)
        self.conv_block[0].bias.data.fill_(0.0)

        std = math.sqrt(2.0 / (32 * 4 * 8 * 8))
        nn.init.normal_(self.conv_block[2].weight, mean=0.0, std=std)
        self.conv_block[2].bias.data.fill_(0.0)

        std = math.sqrt(2.0 / (64 * 32 * 4 * 4))
        nn.init.normal_(self.conv_block[4].weight, mean=0.0, std=std)
        self.conv_block[4].bias.data.fill_(0.0)

        std = math.sqrt(2.0 / (64 * 64 * 3 * 3))
        nn.init.normal_(self.fc[0].weight, mean=0.0, std=std)
        self.fc[0].bias.data.fill_(0.0)


    def forward(self, ob):
        if not isinstance(ob, torch.Tensor):
            ob = torch.from_numpy(ob).float().to(self.device)

        input_dim = ob.dim()

        if input_dim == 3:
            ob = ob.unsqueeze(0)

        cnn_out = self.conv_block(ob)
        flatten_out = torch.flatten(cnn_out, 1)
        fc_out = self.fc(flatten_out)

        V = self.V(fc_out)
        A = self.A(fc_out)

        Q = V + (A - A.mean(dim=-1, keepdim=True))

        if input_dim == 3:
            Q = Q.squeeze(0)

        return Q


    def save(self, path):
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)


    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
