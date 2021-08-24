"""
Data Set을 만드는 여러 방법들을 구현한 모듈.
"""
from utils.marketdata import get_market_data
from utils.share import default_data_path, make_file_name
import numpy as np
import pandas as pd

class Controller:
    def __init__(self):
        self.builder = None

    def construct_dataset(self, builder, **options):
        self.builder = builder(options)
        steps = (builder.get_market_data,
                 builder.partition_market_data,
                 builder.manufacture_pieces,
                 builder.save_result)
        [step(self.builder) for step in steps]

class PastFuture:
    def __init__(self, options:dict) -> None:
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
        self.market_data = get_market_data(self.start_time, self.end_time, self.symbol, self.interval)
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
        return ''.join([func_name,tags,'_',start,'_',end,'_',interval])