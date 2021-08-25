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
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import PastFutureDataset
# pastfuturedata = PastFutureDataset()
# data_loader = torch.utils.data.DataLoader(pastfuturedata,batch_size=1,shuffle=True)

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # First 2D convolutional layer, taking in 1 input channel (image),
      # outputting 32 convolutional features, with a square kernel size of 3
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      # Second 2D convolutional layer, taking in the 32 input layers,
      # outputting 64 convolutional features, with a square kernel size of 3
      self.conv2 = nn.Conv2d(32, 64, 3, 1)

      # Designed to ensure that adjacent pixels are either all 0s or all active
      # with an input probability
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # First fully connected layer
      self.fc1 = nn.Linear(9216, 128)
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(128, 10)

my_nn = Net()
print(my_nn)
print(torch.randn(128, 20))