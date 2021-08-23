"""
Data Set을 만드는 여러 방법들을 구현한 모듈.
"""
from torch.utils import data
from utils.dataset import DataSet
import numpy as np
import pandas as pd
# from exceptions import InvalidDataFrameSizeError

class DataMaker:
    def __init__(self, market_data: pd.DataFrame) :
        self.market_data = market_data
    def make_bundle(self):
        """클래스 생성자의 인수정보를 가지고 데이터셋을 제작,
        데이터셋을 번들형태로 만들어 반환
        """
        pass
    def save(self, path:str, name:str,):
        """클래스를 파일형태로 저장.

        Parameters
        ----------
        path : str
            [description]
        name : str
            [description]
        """
        pass
class PastFutureDataMaker(DataMaker):
    def __init__(
        self, 
        market_data : pd.DataFrame, 
        past_length : int,
        future_length:int
        ):
        """
        Parameters
        ----------
        market_data : pd.DataFrame
        past_length : Int
            과거에 해당되는 데이터의 크기
        future_length : Int
            미래에 해당되는 데이터의 크기
        """
        self.market_data = market_data
        self.past_length = past_length
        self.future_length = future_length
        self.total_length = past_length + future_length

        self.dataset = self.make_bundle()

    def past_future_simple(self, market_data):
        """market_data 과거와 미래로 각각의 length만큼으로 이등분하는 방식
        simple형은 출력을 ['up', 'same', 'down'] 중 하나의 값을 갖도록 하는 형태

        Parameters
        ----------
        market_data : pd.DataFrame


        Returns
        -------
        x : 2-D np.Array
            과거 데이터에 대한 numpy 배열
        y : String
            ['up', 'same', 'down'] 중 하나의 값
        """
        assert len(market_data) >= self.total_length

        #* make x
        x = np.array(market_data[0:self.past_length].to_numpy())
        #* make y
        if market_data.iloc[self.past_length-1]['open'] < market_data.iloc[self.total_length-1]['close']:
            y = 'up'
        elif market_data.iloc[self.past_length-1]['open'] > market_data.iloc[self.total_length-1]['close']:
            y = 'down'
        else:
            y = 'same'

        return x, y

    # override
    def make_bundle(self):
        """
        Returns
        -------
        dataset : DataSet
            DataSet(3-D array, 1-D array)
        """
        bundle_x, bundle_y = [], []
        df = self.market_data
        while(len(df) >= self.total_length):
            x, y = self.past_future_simple(df[len(df)-self.total_length-1:len(df)-1])
            bundle_x.append(x)
            bundle_y.append(y)
            df = df[:len(df)-self.total_length]

        result_x = np.array(bundle_x)
        result_y = np.array(bundle_y)

        return DataSet(result_x, result_y)

    def save(self, path:str = './assets/', name:str = None):
        import pickle
        if name is not None:
            path = ''.join([path,name,'.bin'])
        else:
            start = self.market_data.first_valid_index()
            last = self.market_data.last_valid_index()
            name = ''.join(str(e) for e in ['pastfuture_',self.past_length,'_',self.future_length,'_',start,'_',last])
            path = ''.join([path,name,'.bin'])

        print('PastFuture Dataset is saved ', path)
        with open(path, 'wb') as f:
            pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)

def past_future(
    market_data : pd.DataFrame,
    past_length : int,
    future_length : int
    ) -> DataSet:
    """PAST-FUTURE방식으로 dataset을 만듭니다.
    필수적인 인수로 past_length와 future_length가 있습니다.

    Parameters
    ----------
    market_data : pd.DataFrame
        [description]
    past_length : int
        [description]
    future_length : int
        [description]

    Returns
    -------
    DataSet
        [description]
    """
    total_length = past_length + future_length
    bundle_x, bundle_y = [], []
    size = len(market_data)

    while(size >= total_length):
        x, y = _past_future_piece(market_data[size-total_length-1:size-1],past_length,future_length)
        bundle_x.append(x)
        bundle_y.append(y)
        market_data = market_data[:size-total_length]
        size = len(market_data)
    result_x = np.array(bundle_x)
    result_y = np.array(bundle_y)

    return DataSet(result_x, result_y)

def _past_future_piece(
    market_data_piece : pd.DataFrame,
    pa_len: int,
    fu_len: int
    ):

    """market_data 과거와 미래로 각각의 length만큼으로 이등분하는 방식
    simple형은 출력을 ['up', 'same', 'down'] 중 하나의 값을 갖도록 하는 형태
    Parameters
    ----------
    market_data_piece : pd.DataFrame
        전체 데이터프레임의 일부분을 인수로 받음.
    pa_len : int
        past length
    fu_len : int
        future length
    Returns
    -------
    x : 2-D np.Array
        과거 데이터에 대한 numpy 배열
    y : String
        ['up', 'same', 'down'] 중 하나의 값
    """

    total_length = pa_len + fu_len
    assert len(market_data_piece) >= total_length

    #* make x
    x = np.array(market_data_piece[0:pa_len].to_numpy())

    #* make y
    if market_data_piece.iloc[pa_len-1]['open'] < market_data_piece.iloc[total_length-1]['close']:
        y = 'up'
    elif market_data_piece.iloc[pa_len-1]['open'] > market_data_piece.iloc[total_length-1]['close']:
        y = 'down'
    else:
        y = 'same'

    return x, y



def make_dataset(
    datamaker,
    start_time : str,
    end_time : str,
    symbol : str = 'BTCUSDT',
    interval : str = '1d',
    *args ,
    ) -> None :

    """데이터 수집부터 가공까지 한번에 수행해서 저장까지 하는 메소드.
    모든 데이터셋 파일은 이 함수로 만듬.

    Parameters
    ----------
    start_time : str
        %Y-%M-%D hh:mm:ss 형식 문자열
    end_time : str
        %Y-%M-%D hh:mm:ss 형식 문자열
    interval : str
        간격
    symbol : str
        마켓 종류
    *args : list
        function arguments if it need
    """

    from utils.marketdata import get_market_data
    import pickle
    #get market data
    market_data = get_market_data(start_time=start_time,end_time=end_time,symbol=symbol,interval=interval)
    #make dataset
    dataset = datamaker(market_data, args[0], args[1])
    #make name
    tags = ''.join([str(arg)+':' for arg in args])
    if len(tags) != 0:
        tags = tags[:-1]
    file_name = file_naming(datamaker.__name__,tags,start_time,end_time,interval)
    #save
    path = ''.join(['./assets/',file_name,'.bin'])
    print('PastFuture Dataset is saved ', path)
    with open(path, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def file_naming(func_name:str,tags:str,start:str,end:str,interval:str):
    return ''.join([func_name,tags,'_',start,'_',end,'_',interval])