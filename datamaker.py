"""데이터를 가공해 정적파일로서 만듬
머신러닝의 x, y값으로 바로 쓸 수 있는 번들 형태로 출력
"""

def make_train_bundle(path, dataframe, datamaker):
    """
    trainset 묶음을 파일의 형태로 반환

    Parameters
    ----------
    path : String
        출력물의 저장경로와 파일명
    dataframe : pd.DataFrame
        가공할 대용량의 데이터프레임
    datamaker : Function
        데이터를 만드는데 이용할 함수

    Returns
    -------
    No return
    """
    import numpy as np

    train_x, train_y = make_train_set(dataframe, datamaker)


    pass

def make_train_set(dataframe, preprocessor, datamaker):
    """데이터


    Parameters
    ----------
    dataframe : pandas.DataFrame
        가공할 데이터
    preprocessor : Function
        전처리용 함수
    datamaker : Function
        train_set을 만드는 함수


    Returns
    -------
    x_train : 1-D Array
        학습의 입력값에 사용될 데이터 x값
    y_train : String
        학습의 결괏값에 사용될 데이터 y값
    """

    df = preprocessor(dataframe)
    x_train, y_train = datamaker(df)

    return x_train, y_train