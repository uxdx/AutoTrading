class DataSet:
    def __init__(self, set_x, set_y):
        """
        Parameters
        ----------
        set_x : 2-D np.Array
            print(result_x.shape)

            =>(N, (past_length+future_length)*5)
        set_y : 1-D np.Array
            print(result_y.shape)

            =>(N, )
        """
        self.__set_x = set_x
        self.__set_y = set_y

    @property
    def x(self): #getter
        return self.__set_x
    @property
    def y(self):
        return self.__set_y

    def save(self, path, name):
        """
        Parameters
        ----------
        path : String
            데이터셋을 저장할 폴더의 경로
            ex) 'D:/admin/Documents/'
        name : String
            데이터셋 이름
            ex) 'dataset_01'

        Returns
        -------
        -1 : 실패 시
        0 : 성공 시
        """
        import numpy as np

        path = ''.join([path,name])

        np.savez(path, x=self.__set_x, y=self.__set_x)

        return 0

# class TrainSet(DataSet):
