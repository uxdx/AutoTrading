"""
머신러닝에 필요한 데이터를 제공하는 함수들을 구현
"data" >>(get_data)>> "dataframe" >>(preprocessing)>> "dataset"
    "dataset" >>(get_trainset)>> "train_data, train_results"
    "dataset" >>(get_validationset)>> "validation_data, validation_results"


"""


def get_trainset(method):
    """
    가져온 데이터의 전처리를 해서 머신러닝에 사용할 수 있는 training set을 반환
    params:
        method(str): training set 을 제작하는 방법 algorithm
    returns:
        train_data: 학습의 입력값에 사용될 데이터 x값
        train_results: 학습의 결괏값에 사용될 데이터 label값
    """

    if method == 'some method':
        pass
    elif method == 'another method':
        pass

    return train_data, train_results


def preprocessing(dataFrame, method):
    """

    """
