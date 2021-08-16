"""
머신러닝에 필요한 데이터를 제공하는 함수들을 구현
"data" >>(get_data)>> "dataframe" >>(preprocessing)>> "dataset"
    "dataset" >>(get_trainset)>> "train_data, train_labels"
    "dataset" >>(get_validationset)>> "validation_data, validation_results"


"""
from methods.preprocessor import *
from methods.trainset_maker import *


def make_trainset(dataframe, preprocessor, trainset_maker):
    """
    가져온 데이터를 가공해서 머신러닝에 사용할 수 있는 training set 1개를 반환
    params:
        dataframe(pandas.DataFrame): 가공할 데이터
        preprocessor(Function): 전처리용 함수
        trainset_maker(Function): train_set을 만드는 함수
    returns:
        train_data: 학습의 입력값에 사용될 데이터 x값
        train_labels: 학습의 결괏값에 사용될 데이터 label값
    """

    df = preprocessing(df, preprocessor)
    train_data, train_labels = trainset_maker(df)

    return train_data, train_labels


def preprocessing(dataFrame, method):
    """
    params:
        dataFrame(pandas.DataFrame): 
        method(Function): 데이터 전처리에 사용할 알고리즘
    returns:
        dataFrame(pandas.DataFrame): 전처리가 된 dataframe
    """
    return method(dataFrame)
