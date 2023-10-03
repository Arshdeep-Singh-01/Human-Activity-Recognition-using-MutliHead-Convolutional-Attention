import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModule3(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Conv2d(1, 20, 3) #(1, 90, 3)  --> #(20, 88, 1)
    self.layer2 = nn.Conv2d(20,50,1)  #(20, 88, 1) --> #(50, 88, 1)
    self.fullyConnectedLayer1 = nn.Linear(50*88*1, 4000)
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
    x = x.view(-1, 50*88*1)
    x = F.relu(self.fullyConnectedLayer1(x))
    x = F.relu(self.fullyConnectedLayer2(x))
    return x