import time
import datetime
import numpy as np
import pandas as pd
from IPython.display import display
from get_data import get_market_data, get_history
from methods.preprocessor import normalization_min_max
from methods.trainsetmaker import *


class TestClass(object):
    """
    모든 테스트 클래스는 이 클래스를 상속함.
    모든 테스트 클래스는 self.st 를 이용해서 테스트 시작시간을 구할 수 있음.
    """

    def __init__(self, *args, **kwargs):
        self.st = time.time()


class TestGetData(TestClass):
    def test_get_market_data(self):
        print('Test "get_market_data" method. ')

        startTime = '2021-07-14 00:00:00'
        endTime = '2021-07-16 00:00:00'
        df = get_market_data(
            startTime=startTime, endTime=endTime, symbol='BTCUSDT', interval='1h')
        assert df.shape == np.zeros((48, 5)).shape

        print('Finished test "get_market_data" method. ',
              time.time() - self.st, 'sec')


class TestPreprocessingMethods(TestClass):
    def test_normalization_min_max(self):
        print('Test "normalization_min_max" method. ')

        startTime = '2021-08-14 00:00:00'
        endTime = '2021-08-16 00:00:00'
        df = get_market_data(
            startTime=startTime, endTime=endTime, symbol='BTCUSDT', interval='1h')
        normalized = normalization_min_max(df)

        # print(type(normalized.max()))
        for idx, value in normalized.max().items():
            assert 0.0 <= value <= 1.0
        print('Finished test "normalization_min_max" method. ',
              time.time() - self.st, 'sec')


class TestTrainsetMakerMethods(TestClass):
    def test_Ichimoku_Simple(self):
        print('Test "Ichimoku_Simple" method. ')

        startTime = '2021-08-01 00:00:00'
        endTime = '2021-08-10 00:00:00'
        df = get_market_data(
            startTime=startTime, endTime=endTime, symbol='BTCUSDT', interval='4h')
        train_data, train_label = Ichimoku_Simple(df)

        assert train_data.shape == np.zeros(130,).shape
        assert train_label in ['up', 'same', 'down']

        print('Finished test "normalization_min_max" method. ',
              time.time() - self.st, 'sec')


if __name__ == '__main__':
    TestGetData().test_get_market_data()
    TestPreprocessingMethods().test_normalization_min_max()
    TestTrainsetMakerMethods().test_Ichimoku_Simple()
