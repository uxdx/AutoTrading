import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data import dataloader
from torch.utils.data.dataset import Dataset
from torchvision.transforms.transforms import ToTensor
from data.datasets import DatasetFactory
from nn.trainer import MNISTTrainer, Trainer
from data.marketdata import MarketDataProvider
import numpy as np
from nn.models import SingleRNN
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    factory = DatasetFactory(make_new=True, to_tensor=False, normalize=True, train=True)
    # writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    train_dataset = factory.get_custom_dataset_1m()
    data:np.ndarray = train_dataset.data
    print(np.sum(np.isnan(data)))
    # print(data)
    # data_loader = data.DataLoader(dataset=train_dataset,batch_size=1,shuffle=True)

    # trainer = Trainer(True)
    # trainer.test()

    # from matplotlib import pyplot as plt
    # for xx, yy in data_loader:
    #     plt.plot(xx[0][0])
    #     plt.show()
    #     plt.close()