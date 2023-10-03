import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleMatrix(nn.Module):
  def __int__(self):
    super().__init__()

  def forward(self, x):
    self.fcl = nn.Linear(3000, 128*30)
    x = self.fcl(x)
    # batch_size = 10
    # number of attention heads = 30
    # sequence len = 1
    # dimention of particular sequence = 128
    x = x.reshape(10, 30, 1, 128)
    return x

class QKVMatrix(nn.Module):
  def __init__(self):
    super().__init__()
    self.QMatrix = SingleMatrix()
    self.KMatrix = SingleMatrix()
    self.VMatrix = SingleMatrix()

  def forward(self, x):
    q = self.QMatrix(x)
    k = self.KMatrix(x)
    v = self.VMatrix(x)
    return q,k,v