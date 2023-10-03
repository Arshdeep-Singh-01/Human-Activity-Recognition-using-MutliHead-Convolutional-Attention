#importing tensorflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

#importing other modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import torch

# imports from our file structure
from layers.Attention.MutliHeadAttention import MultiheadAttention
from utils.Data.HARDataset import getData

#################################################### setup ###########################################
print("Using torch", torch.__version__)
torch.manual_seed(42)
# device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################### model ############################################
def trainModel(Model, data_loader, loss_module, optimizer, epochs = 10):
  # put model in train mode
  Model.train()

  for epoch in tqdm(range(epochs)):
    for i, (sequence, labels) in enumerate(data_loader):
      sequence = sequence.to(device)
      labels = labels.to(device).long() 

      # Forward Pass
      preds = model(sequence)
      # preds = preds.squeeze(dim = 1)
      loss = loss_module(preds, labels)

      # Backwards and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (i+1)%5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')


if __name__ == "__main__":
    # intializing model, loss and optimizer
    model = MultiheadAttention(3000)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    # getting dataloaders
    train_data_loader, test_data_loader = getData()
    
    # train model
    trainModel(model, train_data_loader, loss, optimizer, 20000)

    # saving model
    state_dict = model.state_dict()
    torch.save(state_dict, 'C:/Users/Admin/Desktop/CNN_HAR/Code/utils/SavedModels/' + 'mainmodel.tar')






