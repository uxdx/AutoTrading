import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data import dataloader
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
    torch.set_default_dtype(torch.float32)
    train_dataset = CustomDataset(make_new=False, normalize=True, to_tensor=True, train=True)
    print(train_dataset.data.shape)
    print(train_dataset.targets.shape)
    print(train_dataset.targets.max())
    print(train_dataset.targets.min())

    data_loader = data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

    net = nn.Linear(52, 25)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    losses = []
    for epoc in range(100):
        batch_loss = 0.0
        for xx, yy in data_loader:
            xx = torch.reshape(xx, (xx.shape[0], 52))
            optimizer.zero_grad()
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        print(batch_loss)
        losses.append(batch_loss)
    from matplotlib import pyplot as plt
    plt.plot(losses)
    plt.show()