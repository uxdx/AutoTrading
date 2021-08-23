from utils.dataset import DataSet
import pickle
class DataSetLoader:
    def load(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)

        assert type(data) == type(DataSet())