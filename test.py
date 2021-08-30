from nn.trainer import MNISTTrainer
from data.marketdata import MarketDataProvider
import numpy as np

if __name__ == '__main__':
    data = np.empty([0, 2, 26])
    nparray = np.empty([0,26])
    provider = MarketDataProvider()
    dataframe = provider.request_data(label='open')
    print(dataframe)
    array = dataframe.iloc[0:26].to_numpy().reshape(1,26)
    nparray = np.append(nparray, array, axis=0)
    array2 = np.copy(array)
    nparray = np.append(nparray, array2, axis=0)
    print(array)
    print(array.shape)

    nparray = nparray.reshape(1,2,26)
    print(nparray.shape)
    data = np.append(data, nparray, axis=0)
    print(data)
    print(data.shape)