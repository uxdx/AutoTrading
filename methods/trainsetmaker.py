"""
train set 1개를 만드는 알고리즘에 대한 구현
"""
import numpy as np
from exceptions import InvalidDataFrameSizeError


def ichimoku_simple(dataframe):
    """
    일목산인 이론을 참고한 dataframe 분류 알고리즘
    의 간소화 버전
    dataframe을 시간순서대로 26:9로 분류 후
    train_data는 26개의 행을 flatten한 26*5=130 사이즈 배열
    train_label는 9개 행의 처음과 끝만을 비교해서 ['up', 'same', 'down'] 중 하나의 문자열

    params:
        dataframe(pandas.DataFrame):
    returns:
        train_data(np.array): 26*5=130x1 사이즈의 배열
        train_label(str):
    """
    if len(dataframe) > 35:
        # dataframe 사이즈가 35(26+9)보다 클 경우, 오래된 행은 삭제
        dataframe = dataframe[len(dataframe)-36:len(dataframe)-1]
        train_data, train_label = ichimoku_simple(dataframe)
    elif len(dataframe) < 35:
        raise InvalidDataFrameSizeError()
    else:
        train_data = np.array(dataframe[0:26].to_numpy().flatten())
        if dataframe.iloc[25]['open'] < dataframe.iloc[34]['close']:
            train_label = 'up'
        elif dataframe.iloc[25]['open'] > dataframe.iloc[34]['close']:
            train_label = 'down'
        else:
            train_label = 'same'

    return train_data, train_label
