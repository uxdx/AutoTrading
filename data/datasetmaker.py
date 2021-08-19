"""
Data Set을 만드는 여러 방법들을 구현한 모듈.
"""
from data.dataset import DataSet
import numpy as np
# from exceptions import InvalidDataFrameSizeError

class DataMaker:
    def __init__(self, data_frame) :
        self.data_frame = data_frame
    def make_bundle(self):
        """클래스 생성자의 인수정보를 가지고 데이터셋을 제작,
        데이터셋을 번들형태로 만들어 반환
        """
        pass
class PastFutureDataMaker(DataMaker):
    def __init__(self, data_frame, past_length, future_length):
        """
        Parameters
        ----------
        data_frame : pd.DataFrame
        past_length : Int
            과거에 해당되는 데이터의 크기
        future_length : Int
            미래에 해당되는 데이터의 크기
        """
        self.data_frame = data_frame
        self.past_length = past_length
        self.future_length = future_length
        self.total_length = past_length + future_length

    def past_future_simple(self, dataframe):
        """dataframe을 과거와 미래로 각각의 length만큼으로 이등분하는 방식
        simple형은 출력을 ['up', 'same', 'down'] 중 하나의 값을 갖도록 하는 형태

        Parameters
        ----------
        dataframe : pd.DataFrame


        Returns
        -------
        x : 2-D np.Array
            과거 데이터에 대한 numpy 배열
        y : String
            ['up', 'same', 'down'] 중 하나의 값
        """
        assert len(dataframe) >= self.total_length

        #* make x
        x = np.array(dataframe[0:self.past_length].to_numpy())
        #* make y
        if dataframe.iloc[self.past_length-1]['open'] < dataframe.iloc[self.total_length-1]['close']:
            y = 'up'
        elif dataframe.iloc[self.past_length-1]['open'] > dataframe.iloc[self.total_length-1]['close']:
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
        df = self.data_frame
        while(len(df) >= self.total_length):
            x, y = self.past_future_simple(df[len(df)-self.total_length-1:len(df)-1])
            bundle_x.append(x)
            bundle_y.append(y)
            df = df[:len(df)-self.total_length]

        result_x = np.array(bundle_x)
        result_y = np.array(bundle_y)

        return DataSet(result_x, result_y)

