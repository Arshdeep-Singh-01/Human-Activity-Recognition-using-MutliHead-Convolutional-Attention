import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Attention.QKVmatrix import QKVMatrix

from layers.MultiHeadCNN.MutliHeadCNN import ParallelCNN

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, dk = 128, num_heads=30):
        super().__init__()
        #modules
        self.parallelCNN = ParallelCNN()
        self.qkv = QKVMatrix()

        self.dModel = dk*num_heads
        self.num_heads = num_heads
        self.head_dim = 128
        self.fullyConnectedLayer1 = nn.Linear(128*30, 128)
        self.fullyConnectedLayer2 = nn.Linear(128, 6)

        self._reset_parameters()

    def _reset_parameters(self):
        #initialization
        nn.init.xavier_uniform_(self.fullyConnectedLayer1.weight)
        self.fullyConnectedLayer1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fullyConnectedLayer2.weight)
        self.fullyConnectedLayer2.bias.data.fill_(0)

    def forward(self, x):
        x = self.parallelCNN(x)
        q, k, v = self.qkv(x)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        batch_size = 10
        values = values.reshape(batch_size, 128*30) # concatination
        output = F.relu(self.fullyConnectedLayer1(values)) # fully connected layer
        output = self.fullyConnectedLayer2(output)
        output = F.softmax(output, dim=-1)
        return output