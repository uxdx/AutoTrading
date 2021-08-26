"""

Channel = [RSI, MA, CCI, VOLUME, ...]
X.shape = (Batch_size, Channel, Past_length)

Y = (변동률%) = ['-10%~-7%', ... , '-3%~-1%', '-1%~0%', '0%~1%', ...]
목표로하는 Y의 형태 = [0.001, 0.002, 0.004, ..., 0.21, 0.24, 0.34, 0.21, 0.14, ... , 0.001]
이런식으로 정규분포처럼 가운데가 뭉툭한 종 모양

시계열 분석?
LSTM? RNN?

정확도 6할이 목표임.

NETWORK:
    Loss_Func:
        예측값이 실제값과 차이가 있으면 차이가 난 정도에 비례해서 손실함수가 커지지만,
        1~2단계 차이정도라면 함수값이 너무 커서는 안됨.




"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets import PastFutureDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device
# print(torch.cuda.get_device_name(0))
class Simulator:
    def __init__(self) -> None:
        self.dataframe = None
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
        self.network = None
        self.loss_function = None
        self.optimizer = None

    def simulate(self):
        self.data_setting()
        self.parameters_setting()
        self.functions_setting()
        self.learning()
        self.predict()

    def data_setting(self):
        from utils.marketdata import MarketDataProvider
        self.X = None
        self.y = None

        # Train Data
        X_train = None
        X_test = None

        y_train = None
        y_test = None
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
        self.network = None
        self.loss_function = None
        self.optimizer = None

    def learning(self):
        ### 학습
        for epoch in range(self.hyper_parameters['num_epochs']):
            outputs = self.network.forward(self.X_train_tensors.to(device)) #forward pass
            self.optimizer.zero_grad() #caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = self.loss_function(outputs, self.y_train_tensors.to(device))
            loss.backward() #calculates the loss of the loss function

            self.optimizer.step() #improve from loss, i.e backprop
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    def predict(self):
        # 예측
        pass
if __name__ == '__main__':
    Simulator().simulate()



