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
from utils.datasetloader import DataSetLoader

class PastFuture(Dataset):
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
        chart_file = f"[파일명]"
        data = read_chart_file(os.path.join(self.raw_folder, chart_file))

        label_file = f"[파일명]"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

        

    def __len__(self):
        return len(self.img_labels)

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

def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_chart_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x