"""
PyTorch의 Dataset을 상속한 서브 클래스들을 보관.
각 클래스는 파일형태의 데이터에서 데이터셋을 만들어 반환.

End Point Usage Example
---------------
from THIS import datasets

binance_data = datasets.PastFuture(symbol='BTCUSDT',interval='1d',\
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

class CustomDataset(Dataset):
    """여러채널(price,volume,기타 보조지표들 등)을 이용해서
    (N, C, lp) 형태의 data를 만들고,
        N: 데이터셋 크기
        C: 채널 개수
        lp: 과거 시계열의 길이
    
    (N, 2lf+1) 형태의 targets를 만든다.
    2lf+1은 현재 대비 미래의 어느 날짜에 어떤 매매를 할 지 경우의 수를 나타낸다.
    가령, 미래 시계열이 5이면 target의 각 인덱스는 
    5 단위시간 뒤 Long, 4 단위시간 뒤 Long, ..., 무포지션, 1 단위시간 뒤 Short, 2 단위시간 뒤 Short, ..., 5 단위시간 뒤 Short
    이런 의미를 부여받으며,
    그 값은 인덱스대로 매매를 했을 시 이득의 정도를 표현한다.
    가령 실제 데이터가 3 단위시간 뒤 Long포지션을 취했을 때 최대 이익을 얻는다고 가정하면
    target 이런 식으로 될 것이다.
    target = [0.3, 0.4, 0.7, 0.4, 0.2, 0.0, -0.1, -0.15, -0.3, -0.2, -0.2]
    음수의 값은 손해를 보는 것을 의미한다.

    순전파 시 계산되는 y의 값도 마찬가지로 [-1:1] 범위를 갖는 동일 형태이다.
        lf: 미래 시계열의 길이
        y = []
    
    """
    def __init__(self, make_new:bool=False) -> None:
        super().__init__()
        self.pa_len = 26
        self.fu_len = 9
        self.channel_size = 2
        self.make_new = make_new

        self.data = np.empty([0,self.channel_size, self.pa_len]) # (N, 2, 26)
        self.targets = np.empty([0,self.fu_len-1]) # (N, 9-1)
        self.dataframes:list[pd.DataFrame] = []
        self.load_dataset()

    def load_dataset(self):
        def make_dataset():
            """데이터로 데이터셋을 만듬
            """
            def load_data():
                """각 채널에 해당하는 데이터들을 수집
                데이터들의 타입: dataframe
                """
                self.dataframes = [
                    MarketDataProvider().request_data(label='open'),
                    MarketDataProvider().request_data(label='volume')
                ]
            def make_data(idx:int):
                """data(X에 해당)를 만듬
                """
                nparray = np.empty([0,self.pa_len])
                for dataframe in self.dataframes:
                    nparray = np.append(nparray, dataframe.iloc[idx:idx+self.pa_len].to_numpy().reshape(1,26), axis=0) # (1, pa_len)
                nparray = nparray.reshape(1,self.channel_size,self.pa_len) # (C, pa_len) -> (1, C, pa_len)
                self.data = np.append(self.data, nparray, axis=0)

            def make_targets(idx):
                """targets(Y에 해당)을 만듬
                """
                total_len = self.pa_len + self.fu_len
                array = self.dataframes[0].iloc[idx+self.pa_len:idx+total_len].to_numpy()
                current = array[0]
                array = array[1:]
                array = (array-current)/current
                self.targets = np.append(self.targets,array.reshape(1,self.fu_len-1), axis=0) # (1, self.fu_len-1)

            load_data()
            idx = 0
            while idx + self.pa_len + self.fu_len <= len(self.dataframes[0]):
                make_data(idx)
                make_targets(idx)

                if idx % 1000 == 0:
                    print(idx,'/',len(self.dataframes[0]))
                idx += 1
            print('Make data set!')
        def save_as_file():
            np.savez_compressed('./assets/CustomDataset',data=self.data,targets=self.targets)
            print('Dataset Saved.')
        def load_as_file():
            loaded = np.load('./assets/CustomDataset.npz')
            self.data = loaded['data']
            self.targets = loaded['targets']
            assert len(self.targets) == len(self.data)
            print('Dataset Loaded.')

        if self.make_new:
            make_dataset()
            save_as_file()
        else:
            load_as_file()




    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index) -> Tuple[Any, Any]:
        return self.data[index], self.targets[index]