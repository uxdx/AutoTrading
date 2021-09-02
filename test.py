import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data import dataloader
from torchvision.transforms.transforms import ToTensor
from data.datasets import CustomDataset
from nn.trainer import MNISTTrainer, Trainer
from data.marketdata import MarketDataProvider
import numpy as np
from nn.models import SingleRNN

if __name__ == '__main__':
    # arr = np.random.randn(10)
    # print(arr)
    # first = arr[0]]
    # arr = (arr- first)/first
    # print(arr)
    # arr = arr.reshape(1, 10)
    # print(arr.shape)
    # torch.set_default_dtype(torch.float32)
    # train_dataset = CustomDataset(make_new=False, normalize=True, to_tensor=True, train=True)
    # print(train_dataset.data.shape)
    # print(train_dataset.targets.shape)
    # print(train_dataset.targets.max())
    # print(train_dataset.targets.min())




    # data_loader = data.DataLoader(dataset=train_dataset,batch_size=1,shuffle=True)

    trainer = Trainer(True)
    trainer.test()

    # from matplotlib import pyplot as plt
    # for xx, yy in data_loader:
    #     plt.plot(xx[0][0])
    #     plt.show()
    #     plt.close()