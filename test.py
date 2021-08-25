"""모듈 및 함수 검증을 위한 테스트 코드
"""
import time
import numpy as np
import pandas as pd
from requests.api import options
# from utils.marketdata import get_market_data
from utils.datasetmaker import Controller, PastFuture
from utils.marketdata import Binance
from datasets import PastFutureDataset
from typing import List

# Decorator
class Tester:
    def __init__(self, f):
        self.func = f
        self.test_start_time = time.time()
    def __call__(self, *args, **kwargs):
        self.init_msg()
        self.func(self, *args, **kwargs)
        self.ending_msg()
    def init_msg(self):
        print('Start "', self.func.__name__,'". ')
    def ending_msg(self):
        print('Finished "', self.func.__name__, '". ', time.time() - self.test_start_time, 'sec')


@Tester
def get_data_tester(self):
    start_time = '2021-01-14 00:00:00'
    end_time = '2021-07-16 00:00:00'
    binance = Binance()
    dataframe = binance.get_data(start_time,end_time)
    print(dataframe)

if __name__ == '__main__':
    # Tester().execute()
    get_data_tester()
