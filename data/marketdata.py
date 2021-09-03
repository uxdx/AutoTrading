"""마켓(binance)에서 데이터를 얻어오는 함수를 구현.
"""
from numpy import longlong
import pandas as pd
from data.share import datetime_to_unixtime, to_thousands

class MarketDataProvider:
    """
    Usage
    -----
    1) provider = MarketDataProvider('2021-01-01 00:00:00', '2021-05-01 00:00:00','Binance')
    2) market_data = provider.request_data('price') or provider.request_data('volume')
    """
    def __init__(self, symbol, interval, start_time:str=None, end_time:str=None, market_name:str='Binance') -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.market:Market = None

        if market_name == 'Binance':
            self.market = Binance(symbol=symbol ,interval=interval)
        elif market_name == 'ByBit':
            pass

    def request_data(self, label='price') -> pd.DataFrame:
        return self.market.get_data(self.start_time, self.end_time, label)
        #...

class Market:
    def __init__(self) -> None:
        pass
    def get_data(start_str:str,end_str:str) -> pd.DataFrame:
        pass

class Binance(Market):
    def __init__(self, symbol:str='BTCUSDT', interval:str='1h') -> None:
        self.symbol = symbol
        self.interval = interval

        self.dataframe = None
        self.load_dataframe()
    def get_data(self,start_str:str,end_str:str,label:str) -> pd.DataFrame:
        """Get Data from index and label
        """
        if start_str is None and end_str is None:
            data = self.dataframe
        else:
            start_unix = self.dataframe.index[0] if start_str is None else to_thousands(datetime_to_unixtime(start_str))
            end_unix = self.dataframe.index[-1] if end_str is None else to_thousands(datetime_to_unixtime(end_str))
            data = self._slice_data_from_index(self.dataframe,start_unix, end_unix)
        data = self._slice_data_from_label(data, label)
        return data
    def _slice_data_from_index(self, df:pd.DataFrame,start_unix:int,end_unix:int) -> pd.DataFrame:
        return df.loc[start_unix:end_unix].copy()
    def _slice_data_from_label(self, df:pd.DataFrame, label:str):
        if label == 'price':
            return df[['open', 'high', 'low', 'close']]
        elif label == 'open':
            return df['open']
        elif label == 'high':
            return df['high']
        elif label == 'low':
            return df['low']
        elif label == 'close':
            return df['close']
        elif label == 'volume':
            return df['Volume USDT']
        else:
            raise NameError(label)
    def load_dataframe(self) -> None:
        self._read_csv_file() #1
        self._type_setting() #2
        self._set_index_as_unix() #3
        self._reverse() #4
        assert self.dataframe is not None
    def _read_csv_file(self) -> None: #1
        try:
            self.dataframe = pd.read_csv('./assets/Binance_{}_{}.csv'.format(self.symbol, self.interval), usecols=['unix', 'open', 'high', 'low', 'close', 'Volume USDT'])
        except FileNotFoundError:
            print("File not found.", './assets/Binance_{}_{}.csv'.format(self.symbol, self.interval))
            import sys
            sys.exit(0)
    def _type_setting(self) -> None: #2
        self.dataframe = self.dataframe.astype('float32')
        # print(self.dataframe.isnull().values.any())
        self.dataframe = self.dataframe.astype({'unix':longlong}) # float to int (소수점 없애고 정수화)
    def _set_index_as_unix(self) -> None: #3
        self.dataframe = self.dataframe.set_index('unix')
    def _reverse(self) -> None: #4
        self.dataframe = self.dataframe[::-1].copy()