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
from numpy import ndarray
from nn.models import SingleRNN
from torch.utils.tensorboard import SummaryWriter

def get_answer(y:ndarray):
    max = np.max(y)
    min = np.min(y)
    return np.argmax(y) if abs(max) >= abs(min) else np.argmin(y)
if __name__ == '__main__':
    trainer = Trainer(make_new=True)
    trainer.test(plot=True)

    # from matplotlib import pyplot as plt
    # for xx, yy in data_loader:
    #     plt.plot(xx[0][0])
    #     plt.show()

    #     plt.close()