"""
PyTorch의 Dataset을 상속한 서브 클래스들을 보관.
각 클래스는 파일형태의 데이터에서 데이터셋을 만들어 반환.

End Point Usage Example
---------------
from THIS import datasets

binance_data = datasets.PastFuture(symbol='BTCUSDT',interval='1d',
    start='2018-01-01 00:00:00', end='2021-01-01 00:00:00')
"""
import os

from numpy import ndarray
from torch.utils import data
from data.marketdata import MarketDataProvider
import pandas as pd
import numpy as np
import torch
from typing import Any, Callable, List, Optional, Union, Tuple
from torch.utils.data import Dataset


class DatasetFactory:
    """여기서 data와 targets를 만들어 각 데이터셋 객체를 생성하는데 사용
    """
    def __init__(self,make_new:bool, to_tensor:bool, normalize:bool, train:bool) -> None:
        self.make_new = make_new
        self.to_tensor = to_tensor
        self.normalize = normalize
        self.train = train
    def get_custom_dataset_1h(self) -> Dataset:
        return CustomDataset(make_new=self.make_new, to_tensor=self.to_tensor,normalize=self.normalize,train=self.train, interval='1h')
    def get_custom_dataset_1m(self) -> Dataset:
        return CustomDataset(make_new=self.make_new, to_tensor=self.to_tensor,normalize=self.normalize,train=self.train, interval='1m')
class AbstractDataset(Dataset):
    def __init__(self, data, targets, make_new:bool=False, to_tensor:bool=False, normalize:bool=False, train:bool=False) -> None:
        self.data = data
        self.targets = targets

    def __str__(self):
        return ''
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return len(self.targets)
class CustomDataset(AbstractDataset):
    def __init__(self, make_new:bool=False, to_tensor:bool=False, normalize:bool=False, train:bool=False, interval='1h') -> None:
        self.pa_len = 26
        self.fu_len = 26
        self.channel_size = 2
        self.make_new = make_new
        self.interval = interval

        self.data = []
        self.targets = []
        self.dataframes:list[pd.DataFrame] = []
        self.load_dataset()
        self.separate_train_test()

        self.data = self.train_data if train else self.test_data
        self.targets = self.train_targets if train else self.test_targets
        print(self.data[0])
        if normalize:
            self.data_normalization()
        if to_tensor:
            self.data = torch.from_numpy(self.data).float()
            self.targets = torch.from_numpy(self.targets).float()
    def load_dataset(self):
        def make_dataset():
            """데이터로 데이터셋을 만듬
            """
            def load_data():
                """각 채널에 해당하는 데이터들을 수집
                데이터들의 타입: dataframe
                """
                self.dataframes = [
                    MarketDataProvider(symbol='BTCUSDT', interval=self.interval).request_data(label='open'),
                    MarketDataProvider(symbol='BTCUSDT', interval=self.interval).request_data(label='volume')
                ]
            def make_data(idx:int):
                """data(X에 해당)를 만듬
                """
                # nparray = np.empty([0,self.pa_len])
                list = []
                for dataframe in self.dataframes:
                    list.append(dataframe.iloc[idx:idx+self.pa_len].to_numpy())
                self.data.append(list)

            def make_targets(idx):
                """targets(Y에 해당)을 만듬
                """
                total_len = self.pa_len + self.fu_len
                array = self.dataframes[0].iloc[idx+self.pa_len:idx+total_len].to_numpy()
                current = array[0]
                array = array[1:]
                array = (array-current)/current
                self.targets.append(array)

            load_data()
            idx = 0
            while idx + self.pa_len + self.fu_len <= len(self.dataframes[0]):
                make_data(idx)
                make_targets(idx)

                if idx % 1000 == 0:
                    print(idx,'/',len(self.dataframes[0]))
                idx += 1
            # list -> ndarray
            self.data = np.array(self.data)
            self.targets = np.array(self.targets)
            print('Make data set!')
        def save_as_file():
            np.savez_compressed('./assets/{}_{}'.format(self.__class__.__name__, self.interval),data=self.data,targets=self.targets)
            print('Dataset Saved.')
        def load_as_file():
            loaded = np.load('./assets/{}_{}.npz'.format(self.__class__.__name__, self.interval))
            self.data = loaded['data']
            self.targets = loaded['targets']
            assert len(self.targets) == len(self.data)
            print('Dataset Loaded.')

        if self.make_new:
            make_dataset()
            save_as_file()
        else:
            load_as_file()
    def data_normalization(self):
        assert self.data is not None
        for i in range(self.__len__()):
            for j in range(self.channel_size):
                arr:ndarray = self.data[i,j,:]
                mean = arr.mean()
                std = arr.std()
                if std == 0:
                    print(i,',',j)
                    print(self.data[i,j,:])
                assert std != 0
                arr = (arr - mean)/std
                self.data[i,j,:] = arr
    def separate_train_test(self):
        assert self.data is not None
        assert self.targets is not None
        testset_ratio = 0.2
        separate_line = int(self.__len__()* (1-testset_ratio))
        self.train_data = self.data[:separate_line]
        self.train_targets = self.targets[:separate_line]
        self.test_data = self.data[separate_line:]
        self.test_targets = self.targets[separate_line:]