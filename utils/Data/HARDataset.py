from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from utils.Data.frames import create_sequences
from utils.Data.loadNprocessData import preprocessData, readRAWData
from torch.utils.data import DataLoader

batch_size = 10

class HARDataset(torch.utils.data.Dataset):
  def __init__(self, seq, labels):
    super().__init__()
    self.data = torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(1)
    self.label = torch.tensor(np.array(labels))
    self.size = self.__len__()
    self.shape = self.data.shape

  def __len__(self):
        # Number of data point we have.
        return self.data.shape[0]

  def __getitem__(self, idx):
      # Return the idx-th data point of the dataset
      # If we have multiple things to return (data point and label), we can return them as tuple
      data_point = self.data[idx]
      data_label = self.label[idx]
      return data_point, data_label
  
def save_seq_labels(sequences, labels):
    df = pd.DataFrame({'data': sequences, 'labels': labels})
    path = 'C:/Users/Admin/Desktop/CNN_HAR/Code/utils/Data/Dataset/seq_label_save.csv'
    df.to_csv(path)
  
def getData():
    data = readRAWData()
    scaled_data = preprocessData(data)
    # scaled_data = pd.read_csv('./Dataset/WISDM_ar_v1.1/ScaledData.csv')
    sequences, labels = create_sequences(scaled_data, window_size = 90)

    save_seq_labels(sequences, labels)

    train_data, test_data, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    
    train_dataset = HARDataset(train_data, train_labels)
    test_dataset = HARDataset(test_data, test_labels)

    print(train_dataset.shape)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_data_loader, test_data_loader

if __name__ == "__main__":
    traindl, testdl = getData()
    print(len(traindl))
    print(len(testdl))