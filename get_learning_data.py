"""
머신러닝에 필요한 데이터를 제공하는 함수들을 구현
"data" >>(get_data)>> "dataframe" >>(preprocessing)>> "dataset"
    "dataset" >>(get_trainset)>> "train_data, train_label"
    "dataset" >>(get_validationset)>> "validation_data, validation_results"
"""
from methods.preprocessor import *
from methods.trainsetmaker import *
def make_trainset_bundle(dataframe, preprocessor, trainset_maker, bundle_size):
    """
    trainset 묶음을 반환
    
    """
    pass

def make_trainset(dataframe, preprocessor, trainset_maker):
    """가져온 데이터를 가공해서 머신러닝에 사용할 수 있는 training set 1개를 반환


    Parameters
    ----------
    dataframe : pandas.DataFrame
        가공할 데이터
    preprocessor : Function
        전처리용 함수
    trainset_maker : Function
        train_set을 만드는 함수


    Returns
    -------
    train_data : 1-D Array
        학습의 입력값에 사용될 데이터 x값
    train_label : String
        학습의 결괏값에 사용될 데이터 label값
    """

    df = preprocessor(dataframe)
    train_data, train_label = trainset_maker(df)

    return train_data, train_label