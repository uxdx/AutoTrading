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
from utils.marketdata import MarketDataProvider
import pandas as pd
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

class PastFutureDataset(Dataset):
    def __init__(
            self,
            symbol: str = 'BTCUSDT',
            interval: str = '1d',
            start: str = '2018-01-01 00:00:00',
            end: str = '2021-01-01 00:00:00',
            past_length: int = 10,
            future_length: int = 5,
            transform: Optional[Callable] = None,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        self.past_length = past_length
        self.future_length = future_length
        self.transform = transform
        self._arguments_checker()

        self.data, self.targets = self._load_data()
    
    def _arguments_checker(self):
        pass

    def _load_data(self):
        import pickle
        try:
            with open(self._get_data_path(isData=True), 'rb') as f:
                data = pickle.load(f)
            with open(self._get_data_path(isData=False), 'rb') as f:
                targets = pickle.load(f)
        except FileNotFoundError:
            print('File not Found. ', self._get_data_path)
        return data, targets

    def _get_data_path(self, isData:bool):
        return ''.join([
            './assets/',
            'Data_'if isData else 'Targets_',
            'PastFuture',
            self._get_tags(),
            '_',
            self.start,
            '~',
            self.end,
            '_',
            self.interval,
            '.bin'])
    def _get_tags(self):
        return ''.join([str(self.past_length),':',str(self.future_length)])
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (chart, target) where the type of target specified by target_type.
        """
        chart, target = self.data[index], self.targets[index]
        if self.transform is not None:
            chart = self.transform(chart)
        return chart, target

class CustomDataset(Dataset):
    """
    BTCUSDT 1시간봉 모든 시간으로 만든 데이터셋
    slicing 방식
    """
    def __init__(self) -> None:
        super().__init__()
        self.pa_len = 10 # 과거 시계열 길이
        self.fu_len = 5 # 미래 시계열 길이
        self.stride = 1

        self.data:list[pd.DataFrame] = None
        self.targets:list[str] = None
        self.load_dataset()
        assert self.data is not None
        assert self.targets is not None

    def load_dataset(self, make_new:bool=True):
        if make_new:
            self._make_dataset(self._load_marketdata())
        else:
            pass
    def _load_marketdata(self):
        return MarketDataProvider().request_data()
    def _make_dataset(self, market_df:pd.DataFrame):
        X, y = [], []
        idx = 0
        index_max = len(market_df) - (self.pa_len+self.fu_len -1)
        while idx < index_max:
            new_x, new_y = self._slicing(market_df, idx, self.pa_len, self.fu_len)
            X.append(new_x)
            y.append(self._get_target(new_y))
            idx += 1
            if idx % 1000 == 0:
                percentage = idx/index_max*100
                print('Making dataset ... ', idx, '/', index_max, '  ', f'{percentage:.2f}', '%')
        self.data, self.targets = X, y
    def _get_target(self, y_frame:pd.DataFrame):
        if y_frame['open'].iloc[0] < y_frame['close'].iloc[-1]:
            return 'up'
        elif y_frame['open'].iloc[0] > y_frame['close'].iloc[-1]:
            return 'down'
        else:
            return 'same'
    def _slicing(self, dataframe:pd.DataFrame, idx:int, x_len:int, y_len:int):
        return dataframe.iloc[idx:idx+x_len, :], dataframe.iloc[idx+x_len:idx+x_len+y_len, :]
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)