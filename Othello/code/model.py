import torch
import torch.nn as nn
import torch.nn.functional as F

class ReversiNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ReversiNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
