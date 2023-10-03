import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModule2(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Conv2d(1, 16, 3) #(1, 90, 3)  --> #(16, 88, 1)
    self.layer2 = nn.Conv2d(16,40,1)  #(16, 88, 1) --> #(40, 88, 1)
    self.fullyConnectedLayer1 = nn.Linear(40*88*1, 4000)
    self.fullyConnectedLayer2 = nn.Linear(4000, 2000)
    self._reset_parameters()

  def _reset_parameters(self):
    # initialization
    nn.init.xavier_uniform_(self.layer1.weight)
    self.layer1.bias.data.fill_(0)
    nn.init.xavier_uniform_(self.layer2.weight)
    self.layer2.bias.data.fill_(0)
    nn.init.xavier_uniform_(self.fullyConnectedLayer1.weight)
    self.fullyConnectedLayer1.bias.data.fill_(0)
    nn.init.xavier_uniform_(self.fullyConnectedLayer2.weight)
    self.fullyConnectedLayer2.bias.data.fill_(0)

  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = x.view(-1, 40*88*1)
    x = F.relu(self.fullyConnectedLayer1(x))
    x = F.relu(self.fullyConnectedLayer2(x))
    return x