import torch
from torch import nn
from torch import optim

class DuelingDQN(nn.Module):
    def __init__(self, ac_dim, lr=1e-2):
        super(DuelingDQN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.A_fc = nn.Sequential(
            nn.Linear(in_features=6*4*64, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(128, ac_dim)
        )
        self.V_fc = nn.Sequential(
            nn.Linear(in_features=6*4*64, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        self.to(self.device)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr
        )


    def forward(self, ob):
        if not isinstance(ob, torch.Tensor):
            ob = torch.from_numpy(ob).float().to(self.device)

        input_dim = ob.dim()
        if input_dim == 3:
            ob = ob.unsqueeze(0)

        cnn_out = self.conv_block(ob)

        flatten_out = torch.flatten(cnn_out, 1)

        V = self.V_fc(flatten_out)
        A = self.A_fc(flatten_out)

        Q = V + (A - A.mean())

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
