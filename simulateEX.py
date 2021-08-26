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
import networks
from torch.autograd import Variable
from datasets import PastFutureDataset
# pastfuturedata = PastFutureDataset()
# data_loader = torch.utils.data.DataLoader(pastfuturedata,batch_size=1,shuffle=True)


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
        self.lstm = None
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
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        provider = MarketDataProvider('2021-01-01 00:00:00', '2021-07-01 00:00:00')
        self.dataframe = provider.request_data()
        provider = MarketDataProvider('2021-01-05 00:00:00', '2021-07-05 00:00:00')
        self.dataframe2 = provider.request_data()
        self.X = self.dataframe.iloc[:,0:3]
        self.y = self.dataframe2.iloc[:,3:4]
        # print(self.X)
        # print(self.y)

        self.mm = MinMaxScaler()
        self.ss = StandardScaler()
        self.X_ss = self.ss.fit_transform(self.X)
        self.y_mm = self.mm.fit_transform(self.y)
        # Train Data
        X_train = self.X_ss[:3800, :]
        X_test = self.X_ss[3800:, :] # Test Data
        """
        ( 굳이 없어도 된다. 하지만 얼마나 예측데이터와 실제 데이터의 정확도를 확인하기 위해
        from sklearn.metrics import accuracy_score 를 통해 정확한 값으로 확인할 수 있다. )
        """
        y_train = self.y_mm[:3800, :]
        y_test = self.y_mm[3800:, :]
        print("Training Shape", X_train.shape, y_train.shape)
        print("Testing Shape", X_test.shape, y_test.shape)
        """
        torch Variable에는 3개의 형태가 있다. data, grad, grad_fn 한 번 구글에 찾아서 공부해보길 바랍니다.
        """
        X_train_tensors = Variable(torch.Tensor(X_train))
        X_test_tensors = Variable(torch.Tensor(X_test))
        self.y_train_tensors = Variable(torch.Tensor(y_train))
        self.y_test_tensors = Variable(torch.Tensor(y_test))
        self.X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
        self.X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))
        print("Training Shape", self.X_train_tensors_final.shape, self.y_train_tensors.shape)
        print("Testing Shape", self.X_test_tensors_final.shape, self.y_test_tensors.shape)




    def parameters_setting(self):
        # 파라미터 구성
        self.hyper_parameters['num_epochs'] = 30000 #1000 epochs
        self.hyper_parameters['learning_rate'] = 0.00001 #0.001 lr

        self.hyper_parameters['input_size'] = 3 #number of features
        self.hyper_parameters['hidden_size'] = 2 #number of features in hidden state
        self.hyper_parameters['num_layers'] = 1 #number of stacked lstm layers

        self.hyper_parameters['num_classes'] = 1 #number of output classes
    def functions_setting(self):
        self.lstm = networks.LSTM(self.hyper_parameters['num_classes'], self.hyper_parameters['input_size'], self.hyper_parameters['hidden_size'], self.hyper_parameters['num_layers'],
            self.X.shape[1]).to(device)
        self.loss_function = torch.nn.MSELoss()    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.hyper_parameters['learning_rate'])  # adam optimizer

    def learning(self):
        ### 학습
        for epoch in range(self.hyper_parameters['num_epochs']):
            outputs = self.lstm.forward(self.X_train_tensors_final.to(device)) #forward pass
            self.optimizer.zero_grad() #caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = self.loss_function(outputs, self.y_train_tensors.to(device))
            loss.backward() #calculates the loss of the loss function

            self.optimizer.step() #improve from loss, i.e backprop
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    def predict(self):
        # 예측
        df_X_ss = self.ss.transform(self.dataframe.iloc[:,0:3])
        df_y_mm = self.mm.transform(self.dataframe.iloc[:,3:4])

        df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
        df_y_mm = Variable(torch.Tensor(df_y_mm))
        #reshaping the dataset
        df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
        train_predict = self.lstm(df_X_ss.to(device))#forward pass
        data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
        dataY_plot = df_y_mm.data.numpy()

        data_predict = self.mm.inverse_transform(data_predict) #reverse transformation
        dataY_plot = self.mm.inverse_transform(dataY_plot)
        plt.figure(figsize=(10,6)) #plotting
        plt.axvline(x=4500, c='r', linestyle='--') #size of the training set

        plt.plot(dataY_plot, label='Actuall Data') #actual plot
        plt.plot(data_predict, label='Predicted Data') #predicted plot
        plt.title('Time-Series Prediction')
        plt.legend()
        plt.show()
if __name__ == '__main__':

    Simulator().simulate()



