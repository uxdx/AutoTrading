"""모듈 및 함수 검증을 위한 테스트 코드
"""
import time
import numpy as np
from getdata import get_market_data
from methods.preprocessor import normalization_min_max
from methods.trainsetmaker import ichimoku_simple


class TestClass:
    """
    모든 테스트 클래스는 이 클래스를 상속함.
    모든 테스트 클래스는 self.test_start_time 를 이용해서 테스트 시작시간을 구할 수 있음.
    """

    def __init__(self):
        self.test_start_time = time.time()


class TestGetData(TestClass):
    """Test Class to test getdata.py module
    """
    def test_get_market_data(self):
        """
        """
        print('Test "get_market_data" method. ')

        start_time = '2021-07-14 00:00:00'
        end_time = '2021-07-16 00:00:00'
        dataframe = get_market_data(
            start_time=start_time, end_time=end_time, symbol='BTCUSDT', interval='1h')
        assert dataframe.shape == np.zeros((48, 5)).shape

        print('Finished test "get_market_data" method. ',
              time.time() - self.test_start_time, 'sec')


class TestPreprocessingMethods(TestClass):
    """Test Class to test preprocessor.py module
    """
    def test_normalization_min_max(self):
        """
        """
        print('Test "normalization_min_max" method. ')

        start_time = '2021-08-14 00:00:00'
        end_time = '2021-08-16 00:00:00'
        dataframe = get_market_data(
            start_time=start_time, end_time=end_time, symbol='BTCUSDT', interval='1h')
        normalized = normalization_min_max(dataframe)

        # print(type(normalized.max()))
        for _, value in normalized.max().items():
            assert 0.0 <= value <= 1.0
        print('Finished test "normalization_min_max" method. ',
              time.time() - self.test_start_time, 'sec')


class TestTrainsetMakerMethods(TestClass):
    """Test Class to test trainsetmaker.py module
    """
    def test_ichimoku_simple(self):
        """
        """
        print('Test "Ichimoku_Simple" method. ')

        start_time = '2021-08-01 00:00:00'
        end_time = '2021-08-10 00:00:00'
        dataframe = get_market_data(
            start_time=start_time, end_time=end_time, symbol='BTCUSDT', interval='4h')
        train_data, train_label = ichimoku_simple(dataframe)

        assert train_data.shape == np.zeros(130,).shape
        assert train_label in ['up', 'same', 'down']

        print('Finished test "normalization_min_max" method. ',
              time.time() - self.test_start_time, 'sec')


if __name__ == '__main__':
    TestGetData().test_get_market_data()
    TestPreprocessingMethods().test_normalization_min_max()
    TestTrainsetMakerMethods().test_ichimoku_simple()
