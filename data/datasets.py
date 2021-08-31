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

class DatasetFactory:
    def __init__(self, what) -> None:
        self.selector = what
    def get_dataset(self):
        dataset = None
        if self.selector == 'test':
            dataset = None
        return dataset

class CustomDataset(Dataset):
    """
    BTCUSDT 1시간봉 모든 시간으로 만든 데이터셋
    slicing 방식
    """
    def __init__(self, make_new:bool=False) -> None:
        super().__init__()
        self.make_new = make_new # 데이터셋을 새로 만들 것인지, 파일을 불러올 것인지
        self.pa_len = 10 # 과거 시계열 길이
        self.fu_len = 5 # 미래 시계열 길이

        self.data:ndarray = np.empty([0,self.pa_len,self.fu_len])
        self.targets:ndarray = np.empty([0])
        self.load_dataset()

    def load_dataset(self):
        if self.make_new:
            self._make_dataset(self._load_marketdata())
            self._save_as_file()
        else:
            self._load_as_file()
    def _load_marketdata(self):
        return MarketDataProvider().request_data()
    def _load_as_file(self):
        import pickle
        with open('./assets/CustomDataset_data.bin','rb') as f:
            self.data = pickle.load(f)
        with open('./assets/CustomDataset_targets.bin','rb') as f:
            self.targets = pickle.load(f)
        print('Success load.')
    def _save_as_file(self):
        import pickle
        with open('./assets/CustomDataset_data.bin','wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
        with open('./assets/CustomDataset_targets.bin','wb') as f:
            pickle.dump(self.targets, f, pickle.HIGHEST_PROTOCOL)
        
    def _make_dataset(self, market_df:pd.DataFrame):
        idx = 0
        index_max = len(market_df) - (self.pa_len+self.fu_len -1)
        while idx < index_max:
            new_x, new_y = self._slicing(market_df, idx, self.pa_len, self.fu_len)
            # print(new_x.to_numpy())
            self.data = np.append(self.data, new_x, axis=0)
            self.targets = np.append(self.targets, self._get_target(new_y))
            idx += 1
            if idx % 1000 == 0:
                percentage = idx/index_max*100
                print('Making dataset ... ', idx, '/', index_max, '  ', f'{percentage:.2f}', '%')
    def _get_target(self, y_frame:pd.DataFrame):
        open = y_frame['open'].iloc[0]
        close =  y_frame['close'].iloc[-1]
        if open < close:
            return close/open
        elif open > close:
            return -1 * close/open
        else:
            return  0
    def _slicing(self, dataframe:pd.DataFrame, idx:int, x_len:int, y_len:int):
        return dataframe.iloc[idx:idx+x_len, :].to_numpy().reshape(1,10,5), dataframe.iloc[idx+x_len:idx+x_len+y_len, :]
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.data[index], self.targets[index]



class CustomDataset2(Dataset):
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
        self.targets = np.empty([0,self.fu_len]) # (N, 9)
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
                array = (array-array[0])/array[0]
                self.targets = np.append(self.targets,array.reshape(1,self.fu_len), axis=0)

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
            np.savez_compressed('./assets/CustomDataset2',data=self.data,targets=self.targets)
            print('Dataset Saved.')
        def load_as_file():
            loaded = np.load('./assets/CustomDataset2.npz')
            self.data = loaded['data']
            self.targets = loaded['targets']
            print('Dataset Loaded.')

        if self.make_new:
            make_dataset()
            save_as_file()
        else:
            load_as_file()




    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index) -> Tuple[Any, Any]:
        return super().__getitem__(index)