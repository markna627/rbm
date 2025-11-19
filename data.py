
import torchvision
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
dataset = torchvision.datasets.MNIST(root = 'data/', download = True)

class BinaryDataset(Dataset):
  '''
  Custom dataset preprocessing digits into float, binarized values.
  '''
  def __init__(self, data, label):
    self.x = data
    self.y = label
    self.x = torch.tensor(((self.x.float()/255.0) > 0.5), dtype = torch.float32)
    self.y = torch.tensor(self.y, dtype = torch.long)
  def __len__(self):
    return len(self.x)
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
    

def load_data():
  '''
  output:
  train_dataloader (DataLoader): returns train_dataloader contains MNIST digit dataset
  val_dataloader (DataLoader): returns val_dataloader contains MNIST digit dataset
  '''
  train_data, test_data = random_split(dataset, [50000, 10000])

  train_data_images = dataset.data[train_data.indices]
  train_data_labels = dataset.targets[train_data.indices]

  test_data_images = dataset.data[test_data.indices]
  test_data_labels = dataset.targets[test_data.indices]


  print(train_data_images.reshape(-1))
  #Train Data
  v_train_data = train_data_images.reshape(50000,-1)
  v_train_label = train_data_labels
  train_dataset = BinaryDataset(v_train_data, v_train_label)
  train_dataloader = DataLoader(train_dataset, batch_size = 500)

  #Test Data
  v_test_data = test_data_images.reshape(10000,-1)
  v_test_label = test_data_labels
  val_dataset = BinaryDataset(v_test_data, v_test_label)
  val_dataloader = DataLoader(val_dataset, batch_size = 500)
  return train_dataloader, val_dataloader

