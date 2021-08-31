from numpy.random.mtrand import random_integers
import torch
from torch.utils import data
from torchvision.transforms.transforms import ToTensor
from data.datasets import CustomDataset
from nn.trainer import MNISTTrainer
from data.marketdata import MarketDataProvider
import numpy as np
from nn.models import SingleRNN

if __name__ == '__main__':
    # arr = np.random.randn(10)
    # print(arr)
    # first = arr[0]
    # arr = (arr- first)/first
    # print(arr)
    # arr = arr.reshape(1, 10)
    # print(arr.shape)
    dataset = CustomDataset(False)
    print(dataset.data.shape)
    print(dataset.targets.shape)
    dataset.targets = (dataset.targets * 100)

    random_index = np.random.choice(dataset.__len__(),size=10, replace=False)
    print(random_index)
    sample_data,sample_targets = dataset.data[random_index], dataset.targets[random_index]
    tensor_data = torch.from_numpy(sample_data).float()
    tensor_targets = torch.from_numpy(sample_targets).float()
    print(tensor_data.shape)
    batch_size, feature_size, input_size = sample_data.shape
    hidden_size = 50
    num_layer = 2
    import torch.nn as nn
    model = nn.LSTM(input_size,hidden_size,num_layer,batch_first=True)
    output, _ = model(tensor_data)
    print(output)