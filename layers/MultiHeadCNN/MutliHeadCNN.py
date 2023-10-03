import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.MultiHeadCNN.CNN.CNNHead1 import CNNModule1
from layers.MultiHeadCNN.CNN.CNNHead2 import CNNModule2
from layers.MultiHeadCNN.CNN.CNNHead3 import CNNModule3

class ParallelCNN(nn.Module):
    def __init__(self):
        super(ParallelCNN, self).__init__()
        self.cnn1 = CNNModule1()
        self.cnn2 = CNNModule2()
        self.cnn3 = CNNModule3()

    def forward(self, x):
        output1 = self.cnn1(x)
        output2 = self.cnn2(x)
        output3 = self.cnn3(x)
        concatenated_output = torch.cat((output1, output2, output3), dim=1)  # Concatenate along the channel dimension
        output = F.max_pool1d(concatenated_output, kernel_size=2, stride=2)
        # print(output.shape)
        return output