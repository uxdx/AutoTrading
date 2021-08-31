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
from nn.models import NeuralNetwork
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device
# print(torch.cuda.get_device_name(0))
class Trainer:
    def __init__(self) -> None:
        self.hyper_parameters = {
            'epochs' : 10000,
            'learning_rate' : 0.001,
            'input_size' : None,
            'hidden_size' : None,
            'num_layers' : 1,
            'num_classes' : 1,

        }
        self.X = None
        self.y = None
        #Functions
        self.model = None
        self.loss_function = None
        self.optimizer = None

    def train(self):
        self.data_setting()
        # self.parameters_setting()
        # self.functions_setting()
        # self.fit()
        # self.plot()

    def data_setting(self):
        dataset = CustomDataset(make_new=False)
        X = dataset.data
        y = dataset.targets

        # Train Data
        X_train = X[:33000]
        X_test = X[33000:]

        y_train = y[:33000]
        y_test = y[33000:]
        print("Training Shape", X_train.shape, y_train.shape)
        print("Testing Shape", X_test.shape, y_test.shape)

        self.y_train_tensors = None
        self.y_test_tensors = None
        print("Training Shape", self.X_train_tensors.shape, self.y_train_tensors.shape)
        print("Testing Shape", self.X_test_tensors.shape, self.y_test_tensors.shape)




    def parameters_setting(self):
        # 파라미터 구성
        self.hyper_parameters['num_epochs'] = 30000 #1000 epochs
        self.hyper_parameters['learning_rate'] = 0.00001 #0.001 lr

        self.hyper_parameters['input_size'] = 3 #number of features
        self.hyper_parameters['hidden_size'] = 2 #number of features in hidden state
        self.hyper_parameters['num_layers'] = 1 #number of stacked lstm layers

        self.hyper_parameters['num_classes'] = 1 #number of output classes
    def functions_setting(self):
        self.model = None
        self.loss_function = None
        self.optimizer = None

    def fit(self):
        ### 학습
        for epoch in range(self.hyper_parameters['num_epochs']):
            outputs = self.model.forward(self.X_train_tensors.to(device)) #forward pass
            self.optimizer.zero_grad() #caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = self.loss_function(outputs, self.y_train_tensors.to(device))
            loss.backward() #calculates the loss of the loss function

            self.optimizer.step() #improve from loss, i.e backprop
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    def plot(self):
        # 학습 결과 출력
        pass

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