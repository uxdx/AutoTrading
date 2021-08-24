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
from utils.dataset import DataSet

class CustomDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.default_path = './assets/'

    def __str__(self) :
        return '(x.shape: {}, y.shape: {})'.format(self.data.shape, self.targets.shape)

    def _load_dataset(self) -> None:
        pass
    def file_naming(self, tags:str,start:str,end:str,interval:str):
        return ''.join([self.__class__.__name__,tags,'_',start,'_',end,'_',interval])

class PastFuture(CustomDataset):
    def __init__(
            self,
            symbol: str = 'BTCUSDT',
            interval: str = '1d',
            start: str = '2018-01-01 00:00:00',
            end: str = '2021-01-01 00:00:00',
            past_length : int = 9,
            future_length : int = 3,
            transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        self.past_length = past_length
        self.future_length = future_length
        self.transform = transform

        self.data, self.targets = self._load_dataset()

    def _load_dataset(self):
        """Loads dataset from file
        """
        dataset =  self._file_load()
        return dataset.x, dataset.y

    def _file_load(self):
        import pickle
        tags = self.get_tags()
        name = self.file_naming(tags, self.start, self.end, self.interval)
        path = ''.join([self.default_path, name,'.bin'])
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            return data
        except FileNotFoundError:
            print(path, ' 파일을 찾을 수 없습니다. 정확한 경로와 이름을 지정해주세요.')
            import sys
            sys.exit(0)

    def get_tags(self):
        return ''.join(str(s) for s in [self.past_length, ':', self.future_length])

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

