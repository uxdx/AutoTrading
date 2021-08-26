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