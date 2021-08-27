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
from utils.marketdata import MarketDataProvider
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

        self.data:ndarray = np.empty([0,10,5])
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