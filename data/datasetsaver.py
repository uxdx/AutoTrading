"""데이터셋을 저장하는 기능
Usage
-----
saver = DataSetSaver()
saver.save("/db", 'dataset_01')
"""
class DataSetSaver:
    def save(self, path, name):
        """
        Parameters
        ----------
        path : String
            데이터셋을 저장할 폴더의 경로
        name : String
            데이터셋 이름

        Returns
        -------
        None
        """
        