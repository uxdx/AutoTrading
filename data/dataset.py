"""DataSet 클래스는 data를 보관하는 클래스

Usage
-----
usage 1 : load()를 이용
    dataset = DataSet()
    dataset.load('.temp/','ds1')

    print(dataset.x)
    print(dataset.y)

usage 2 : 새 데이터를 생성
    dataset = DataSet(x=np.array([[]]),y=np.array([]))
"""
class DataSet:
    def __init__(self, x=None, y=None):
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
        self.__x = x
        self.__y = y
    def __str__(self) :
        return '(x.shape: {}, y.shape: {})'.format(self.x.shape, self.y.shape)
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

    def load(self, path, name, flatten=False, normalize=False, one_hot_incoding=False):
        """파일로부터 데이터셋을 로드

        Parameters
        ----------
        path : String
            불러올 파일의 경로
            ex) './temp'
        name : String
            불러올 파일 이름
            ex) 'ds1'
        flatten : bool
            데이터를 1차원으로 평탄화해서 불러올 지
        normalize : bool
            데이터를 정규화 할지 여부
        one_hot_incoding : bool
            y값을 배열로 표현할 지, 값으로 표현할 지 여부
        """
        import numpy as np

        path = ''.join([path,name,'.npz'])
        try:
            loaded = np.load(path)

            self.__x = loaded['x']
            self.__y = loaded['y']
            loaded.close()
        except FileNotFoundError:
            print(path, ' 파일을 찾을 수 없습니다. 정확한 경로와 이름을 지정해주세요.')
            import sys
            sys.exit(0)

# class TrainSet(DataSet):
