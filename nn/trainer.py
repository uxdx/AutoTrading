"""

Channel = [RSI, MA, CCI, VOLUME, ...]
X.shape = (Batch_size, Channel, Past_length)

Y = (변동률%) = ['-10%~-7%', ... , '-3%~-1%', '-1%~0%', '0%~1%', ...]
목표로하는 Y의 형태 = [0.001, 0.002, 0.004, ..., 0.21, 0.24, 0.34, 0.21, 0.14, ... , 0.001]
이런식으로 정규분포처럼 가운데가 뭉툭한 종 모양

시계열 분석?
LSTM? RNN?

정확도 6할이 목표임.

model:
    Loss_Func:
        예측값이 실제값과 차이가 있으면 차이가 난 정도에 비례해서 손실함수가 커지지만,
        1~2단계 차이정도라면 함수값이 너무 커서는 안됨.


학습시키고, 저장

"""
from torch.nn.modules.linear import Linear
from torch.optim.optimizer import Optimizer
from nn.models import Network, NeuralNetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.autograd import Variable
from data.datasets import CustomDataset
from torch.utils.data import Dataset
from torch import optim


# print(torch.cuda.get_device_name(0))
class Trainer:
    def __init__(self, make_new:bool=False) -> None:
        self.batch_size = 1000
        torch.set_default_dtype(torch.float32)
        #data setting
        train_dataset = CustomDataset(make_new=False, normalize=True, to_tensor=True, train=True)
        test_dataset = CustomDataset(make_new=False, normalize=True, to_tensor=True, train=False)
        self.data_loader_train = DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True,num_workers = 4, pin_memory = True)
        self.data_loader_test = DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=True,num_workers = 4, pin_memory = True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device
        # self.device = torch.device("cpu")
        print(self.device)
        self.model = Linear(52, 25).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        if make_new:
            self.train()
            self.save()
        else:
            self.load()
    def train(self):
        self.losses = []
        epoch = 1000
        for epoc in range(epoch):
            batch_loss = 0.0
            for X, y in self.data_loader_train:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                X = X.reshape(X.shape[0], 52)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.item()
            self.losses.append(batch_loss)
            print(batch_loss)
            print(epoc, '/', epoch)
    def test(self):
        device = 'cpu'
        self.model.eval().to(device)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.data_loader_test:
                X, y = X.to(device), y.to(device)
                X = X.reshape(X.shape[0], 52)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                for i in range(self.batch_size):
                    from matplotlib import pyplot as plt
                    figure, axis = plt.subplots(1, 2)
                    axis[0].plot(range(25), pred[i])
                    axis[0].set_title("prediction")
                    axis[1].plot(range(25), y[i])
                    axis[1].set_title("real")
                    plt.show()
        print(test_loss)
    def plot(self):
        from matplotlib import pyplot as plt
        plt.plot(self.losses)
        plt.show()

    def save(self):
        torch.save(self.model.state_dict(), 'model.pth')
    def load(self):
        self.model.load_state_dict(torch.load('model.pth'))
        print('load succeed')
class MNISTTrainer:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 64
    def fit(self):
        epochs = 5
        for t in range(epochs):
            self.data_setting()
            self.functions_setting()
            self.train(self.train_dataloader)
            self.test(self.test_dataloader)
            self.save()
        print("Done!")
    def data_setting(self):
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        self.train_dataloader = DataLoader(training_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size)
    
    def functions_setting(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = NeuralNetwork()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def train(self, dataloader):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    def save(self):
        torch.save(self.model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")