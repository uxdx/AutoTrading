"""
전처리 방법(알고리즘)에 대한 구현
"""
import pandas as pd


def normalization_min_max(dataFrame):
    """
    데이터의 min-max값을 기준으로 0과 1사이로 정규화 시키는 알고리즘
    z = (x-min)/(max-min)
    데이터프레임의 [open, high, low, close]열에선 중 최솟값을 0으로, 최댓값을 1로 정해
    나머지 값들을 정규화.
    volume열은 따로 정규화
    ex) [0.1, 0.4, 0.1, 0.2, 30]
        [0.2, 0.5, 0.2, 0.3, 25] 이면
        0.1, 25가 0으로, 0.5, 30가 1이 됨

    params:
        dataFrame(pd.DataFrame):
    returns:
        dataFrame(pd.DataFrame):

    """
    def min_max_normalize(x, min, max):
        """
        params:
            x(float): 정규화할 값
            min(float): 정규화에 쓰일 최솟값
            max(float): 정규화에 쓰일 최댓값
        return: 
            (float): (x-min)/(max-min)
        """
        return (x-min)/(max-min)
    # ? min, max 를 구함
    price_max = dataFrame['high'].max()
    price_min = dataFrame['low'].min()
    volume_max = dataFrame['volume'].max()
    volume_min = dataFrame['volume'].min()
    # ? dataFrame을 정규화
    left_df = dataFrame[['open', 'high', 'low', 'close']
                        ].applymap(lambda x: min_max_normalize(x, price_min, price_max))
    right_df = dataFrame[['volume']
                         ].applymap(lambda x: min_max_normalize(x, volume_min, volume_max))

    df = pd.concat([left_df, right_df], axis=1)
    return df


def normalization_z_score(dataFrame):
    """
    데이터의 평균과 표준편차를 이용해서 0과 1사이로 정규화 시키는 알고리즘
    z = (x-mean)/std
    volume열은 따로 정규화

    params:
        dataFrame(pd.DataFrame):
    returns:
        dataFrame(pd.DataFrame):

    """

def identity_function(dataframe):
    """항등함수
    """
    return dataframe