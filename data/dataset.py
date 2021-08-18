class DataSet:
    def __init__(self, set_x=None, set_y=None):
        """
        Parameters
        ----------
        인수는 안쓰던가(load로 내용물을 추가)
        시작부터 x,y다 지정해주던가 해야함.
        set_x : 2-D np.Array
            print(result_x.shape)

            =>(N, (past_length+future_length)*5)
        set_y : 1-D np.Array
            print(result_y.shape)

            =>(N, )
        """
        self.__x = set_x
        self.__y = set_y

    @property
    def x(self): #getter
        return self.__x
    @property
    def y(self):
        return self.__y

    @x.setter
    def x(self, value): #setter
        """set x
        """
        self.__x = value
    @y.setter
    def y(self, value):
        """set y
        """
        self.__y = value


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
        """
        import numpy as np

        path = ''.join([path,name])

        np.savez(path, x=self.x, y=self.y)

    def load(self, path, name):
        """파일로부터 데이터셋을 로드

        Parameters
        ----------
        path : String
            불러올 파일의 경로
            ex) './temp'
        name : String
            불러올 파일 이름
            ex) 'ds1'
        """
        import numpy as np

        path = ''.join([path,name,'.npz'])

        loaded = np.load(path)

        self.__x = loaded['x']
        self.__y = loaded['y']
        loaded.close()

# class TrainSet(DataSet):
