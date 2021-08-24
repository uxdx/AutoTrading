"""
PyTorch의 Dataset을 상속한 서브 클래스들을 보관.
각 클래스는 파일형태의 데이터에서 데이터셋을 만들어 반환.

End Point Usage Example
---------------
from THIS import datasets

binance_data = datasets.PastFuture(symbol='BTCUSDT',interval='1d',\
    start='2018-01-01 00:00:00', end='2021-01-01 00:00:00')
"""
import pickle
from typing import Any, Callable, List, Optional, Union, Tuple
from torch._C import NoneType
from torch.utils.data import Dataset
from utils.share import *

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

class PastFutureDataset(CustomDataset):
    def __init__(
            self,
            normalization:bool = False,
            flatten:bool = False,
            transform:function = None,
            **settings,
    ) -> None:
        super().__init__()
        self.symbol = settings['symbol']
        self.interval = settings['interval']
        self.start = settings['start']
        self.end = settings['end']
        self.past_length = settings['past_length']
        self.future_length = settings['future_length']

        self.data, self.targets = self._load_as_file()

        if normalization:
            pass
        if flatten:
            pass

    def _load_as_file(self):
        return self._load_data(), self._load_targets()

    def _load_data(self):
        tags = self._get_tags()
        name = make_file_name(True,'PastFuture',tags,self.start,self.end,self.interval)
        path = ''.join([default_data_path(), name,'.bin'])
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            return data
        except FileNotFoundError:
            print(path, ' 파일을 찾을 수 없습니다. 정확한 경로와 이름을 지정해주세요.')
            import sys
            sys.exit(0)
    def _load_targets(self):
        tags = self._get_tags()
        name = make_file_name(False,'PastFuture',tags,self.start,self.end,self.interval)
        path = ''.join([default_data_path(), name,'.bin'])
        try:
            with open(path, 'rb') as f:
                targets = pickle.load(f)

            return targets
        except FileNotFoundError:
            print(path, ' 파일을 찾을 수 없습니다. 정확한 경로와 이름을 지정해주세요.')
            import sys
            sys.exit(0)

    def _get_tags(self):
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
