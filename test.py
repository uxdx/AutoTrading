"""모듈 및 함수 검증을 위한 테스트 코드
"""
import time
import numpy as np
import pandas as pd
from requests.api import options
from utils.datasetmaker import Controller, PastFuture, SlicingPF
from utils.marketdata import MarketDataProvider
from datasets import PastFutureDataset
from typing import List

# Decorator
class Tester:
    def __init__(self, f):
        self.func = f
        self.test_start_time = time.time()
    def __call__(self):
        self.init_msg()
        self.func()
        self.ending_msg()
    def init_msg(self):
        print('Start "', self.func.__name__,'". ')
    def ending_msg(self):
        print('Finished "', self.func.__name__, '". ', time.time() - self.test_start_time, 'sec')


@Tester
def get_data_tester():
    start_time = '2021-01-14 00:00:00'
    end_time = '2021-07-16 00:00:00'
    provider = MarketDataProvider(start_time,end_time,'Binance')
    dataframe = provider.request_data()
    print(dataframe)

@Tester
def dataset_maker_tester():
    controller = Controller()
    options = {
        'market':'Binance',
        'start_time' : '2021-07-14 00:00:00',
        'end_time' : '2021-07-16 00:00:00',
        'symbol' : 'BTCUSDT',
        'interval' : '1h',
        'past_length' : 10,
        'future_length' : 5,
        'stride' : 1,
    }
    controller.construct_dataset(SlicingPF, **options)

@Tester
def dataset_loader_tester():
    dataset = PastFutureDataset(symbol='BTCUSDT', interval='1h', start='2021-01-14 00:00:00', end='2021-07-16 00:00:00', past_length=10, future_length=5)
    data = dataset.data
    targets = dataset.targets
    print(data.shape)
    print(targets.shape)

if __name__ == '__main__':
    # Tester().execute()
    # get_data_tester()
    dataset_maker_tester()
    # dataset_loader_tester()
