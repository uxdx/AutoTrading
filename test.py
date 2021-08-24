"""모듈 및 함수 검증을 위한 테스트 코드
"""
import time
import numpy as np
import pandas as pd
from requests.api import options
from utils.marketdata import get_market_data
from utils.datasetmaker import Controller, PastFuture
from datasets import PastFutureDataset
from typing import List
class TestClass:
    """
    모든 테스트 클래스는 이 클래스를 상속함.
    모든 테스트 클래스는 self.test_start_time 를 이용해서 테스트 시작시간을 구할 수 있음.
    """

    def __init__(self):
        self.test_start_time = time.time()

class TestGetData(TestClass):
    def test(self):
        print('Test "get_market_data" method. ')

        start_time = '2021-07-14 00:00:00'
        end_time = '2021-07-16 00:00:00'
        dataframe = get_market_data(
            start_time=start_time, end_time=end_time, symbol='BTCUSDT', interval='30m')
        # pd.set_option('display.max_rows', None)
        # print(dataframe)
        # print(type(dataframe))
        assert dataframe.shape == np.zeros((96, 5)).shape
        # print(dataframe)
        print('Finished test "get_market_data" method. ',
              time.time() - self.test_start_time, 'sec')

class TestMakeDataset(TestClass):
    def test(self):
        print('Test "datasetmaker" module. ')
        start = '2021-07-14 00:00:00'
        end = '2021-08-16 00:00:00'
        options = {'start_time':start, 'end_time':end, 'symbol':'BTCUSDT', 'interval':'1h', 'past_length':10,'future_length':5}
        controller = Controller()
        controller.construct_dataset(builder=PastFuture, **options)
        print('Finished test "datasetmaker" module. ',
              time.time() - self.test_start_time, 'sec')
class TestLoadDataset(TestClass):
    def test(self):
        print('Test "data loading" action. ')
        start_time = '2021-07-14 00:00:00'
        end_time = '2021-08-16 00:00:00'
        # print(sample.SAMPLE_DATAFRAME)
        dataset = PastFutureDataset(start=start_time,end=end_time,symbol='BTCUSDT',interval='1h',past_length=10,future_length=5)
        print(dataset)
        print('Finished test "data loading" action. ',
              time.time() - self.test_start_time, 'sec')

class Tester:
    def __init__(self) -> None:
        self.test_list = List[TestClass]

    def execute(self):
        self.test_list = [TestGetData(),TestMakeDataset(), TestLoadDataset()]
        for test in self.test_list:
            test.test()

if __name__ == '__main__':
    Tester().execute()