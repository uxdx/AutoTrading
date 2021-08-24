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

class PastFutureDataset(Dataset):
    def __init__(
            self,
            symbol: str = 'BTCUSDT',
            interval: str = '1d',
            start: str = '2018-01-01 00:00:00',
            end: str = '2021-01-01 00:00:00',
            transform: Optional[Callable] = None,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        self.transform = transform

        self.data, self.targets = self._load_data()

    def _load_data(self):
        dataset = DataSet.load()
        data = dataset.x
        targets = dataset.y

        return data, targets


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