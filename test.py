from data.datasets import CustomDataset2
from nn.trainer import MNISTTrainer
from data.marketdata import MarketDataProvider
import numpy as np

if __name__ == '__main__':
    # arr = np.random.randn(10)
    # print(arr)
    # first = arr[0]
    # arr = (arr- first)/first
    # print(arr)
    # arr = arr.reshape(1, 10)
    # print(arr.shape)
    dataset = CustomDataset2(False)
    print(dataset.data.shape)
    print(dataset.targets.shape)
    targets = (dataset.targets * 100)
    print(targets.var())
    print(targets.mean())