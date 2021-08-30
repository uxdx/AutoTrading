"""data module tester
"""
import time
import numpy as np
import pandas as pd
from requests.api import options
from data.marketdata import MarketDataProvider
from data.datasets import CustomDataset
from typing import List
import unittest


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

class MarketDataTest(unittest.TestCase):
    def test_datalength(self):
        df1 = MarketDataProvider().request_data()
        self.assertEqual(len(df1), 37881)
        start_time = '2021-01-14 00:00:00'
        end_time = '2021-07-16 00:00:00'
        df2 = MarketDataProvider(start_time,end_time,'Binance').request_data()
        self.assertEqual(len(df2), 4386)
        df3 = MarketDataProvider(start_time=start_time).request_data()
        self.assertEqual(len(df3), 5351)
        df4 = MarketDataProvider(end_time=end_time).request_data()

class DatasetsTest(unittest.TestCase):
    def test_(self):
        pass

unittest.main()