"""
Data Set을 만드는 여러 방법들을 구현한 모듈.
"""
from data.marketdata import MarketDataProvider
from data.share import default_data_path, make_file_name
import numpy as np
import pandas as pd

class Controller:
    """
    Usage
    -----
    1) controller = Controller()
    2) options = {'market': 'Binance, ... ,}
    3) controller.construct_dataset(PastFuture, **options)
    """
    def __init__(self):
        self.builder = None

    def construct_dataset(self, builder, **options):
        self.builder = builder(options)
        steps = (builder.get_market_data,
                 builder.make_dataset,
                 builder.save_result)
        [step(self.builder) for step in steps]

class SlicingPF:
    def __init__(self, options:dict) -> None:
        """
        Options
        -------
        'market' : str = 'Binance'
        'start_time' : str
        'end_time' :
        'symbol' :
        'interval' :
        'stride' : int
        'past_length' :
        'future_length' :
        """
        from data.share import none_init
        self.market = none_init(options, 'market')
        self.start_time = none_init(options, 'start_time')
        self.end_time = none_init(options, 'end_time')
        self.symbol = none_init(options, 'symbol')
        self.interval = none_init(options, 'interval')

        self.stride = options['stride']
        self.past_length = options['past_length']
        self.future_length = options['future_length']
        ##############################################
        self.market_data = pd.DataFrame(None)
        self.data = []
        self.targets = []
    def get_market_data(self):
        provider = MarketDataProvider(self.start_time,self.end_time,self.market)
        self.market_data = provider.request_data() #reverse

    def make_dataset(self, x_len, y_len):
        X, y = [], []
        idx = 0
        while idx < len(self.market_data)-(x_len+y_len -1):
            new_x, new_y = self._slicing(self.market_data, idx, x_len, y_len)
            X.append(new_x)
            y.append(self._get_target(new_y))
            idx += 1
        self.data, self.targets = X, y
    def save_result(self):
        import pickle
        tags = ''.join(str(s) for s in [self.past_length,':',self.future_length])
        file_name = make_file_name(True,self.__class__.__name__,tags,self.start_time,self.end_time,self.interval)
        #save data
        path = ''.join([default_data_path(),file_name,'.bin'])
        print('PastFuture Dataset is saved ', path)
        with open(path, 'wb') as f:
            pickle.dump(np.array(self.data), f, pickle.HIGHEST_PROTOCOL)
        # save targets
        file_name = make_file_name(False,self.__class__.__name__,tags,self.start_time,self.end_time,self.interval)
        path = ''.join([default_data_path(),file_name,'.bin'])
        print('PastFuture Dataset is saved ', path)
        with open(path, 'wb') as f:
            pickle.dump(np.array(self.targets), f, pickle.HIGHEST_PROTOCOL)
#=======Sub Methods==============================
    def _get_target(y_frame:pd.DataFrame):
        if y_frame['open'].iloc[0] < y_frame['close'].iloc[-1]:
            return 'up'
        elif y_frame['open'].iloc[0] > y_frame['close'].iloc[-1]:
            return 'down'
        else:
            return 'same'
    def _slicing(dataframe:pd.DataFrame, idx:int, x_len:int, y_len:int):
        return dataframe.iloc[idx:idx+x_len, :], dataframe.iloc[idx+x_len:idx+x_len+y_len, :]
class PastFuture:
    def __init__(self, options:dict) -> None:
        """
        Options
        -------
        'market' : str = 'Binance'
        'start_time' :
        'end_time' :
        'symbol' :
        'interval' :
        'past_length' :
        'future_length' :
        """
        self.market = options['market']
        self.start_time = options['start_time']
        self.end_time = options['end_time']
        self.symbol = options['symbol']
        self.interval = options['interval']

        self.past_length = options['past_length']
        self.future_length = options['future_length']
        ##############################################
        self.market_data = pd.DataFrame(None)
        self.market_data_pieces = []
        self.data = []
        self.targets = []

    def get_market_data(self):
        provider = MarketDataProvider(self.start_time,self.end_time,self.market)
        self.market_data = provider.request_data()
    def make_dataset(self):
        self.partition_market_data()
        self.manufacture_pieces()
    def partition_market_data(self):
        dataframe = self.market_data
        # save divided dataframe in self.market_data_pieces
        while(len(dataframe)>=self._total_length()):
            self.market_data_pieces.append(dataframe[len(dataframe)-self._total_length()-1:len(dataframe)-1])
            dataframe = dataframe[:len(dataframe)-self._total_length()]
    def manufacture_pieces(self):
        for piece in self.market_data_pieces:
            self.data.append(np.array(piece[0:self.past_length].to_numpy()))
            if piece.iloc[self.past_length-1]['open'] < piece.iloc[self._total_length()-1]['close']:
                y = 'up'
            elif piece.iloc[self.past_length-1]['open'] > piece.iloc[self._total_length()-1]['close']:
                y = 'down'
            else:
                y = 'same'
            self.targets.append(y)
    def save_result(self):
        import pickle
        tags = ''.join(str(s) for s in [self.past_length,':',self.future_length])
        file_name = make_file_name(True,self.__class__.__name__,tags,self.start_time,self.end_time,self.interval)
        #save data
        path = ''.join([default_data_path(),file_name,'.bin'])
        print('PastFuture Dataset is saved ', path)
        with open(path, 'wb') as f:
            pickle.dump(np.array(self.data), f, pickle.HIGHEST_PROTOCOL)
        # save targets
        file_name = make_file_name(False,self.__class__.__name__,tags,self.start_time,self.end_time,self.interval)
        path = ''.join([default_data_path(),file_name,'.bin'])
        print('PastFuture Dataset is saved ', path)
        with open(path, 'wb') as f:
            pickle.dump(np.array(self.targets), f, pickle.HIGHEST_PROTOCOL)
    def _total_length(self):
        return self.past_length + self.future_length