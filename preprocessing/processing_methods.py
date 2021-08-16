"""
전처리 방법(알고리즘)에 대한 구현
"""


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


def normalization_z_score(dataFrame):
    """
    데이터의 평균과 표준편차를 이용해서 0과 1사이로 정규화 시키는 알고리즘
    z = (x-mean)/std
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
