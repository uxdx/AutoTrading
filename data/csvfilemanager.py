import pandas as pd
class CSVManager:
    def __init__(self) -> None:
        self.dataframes = []
        self.load_csv()
    def load_csv(self):
        # dataframe setting
        self.dataframe = pd.read_csv('./assets/Binance_BTCUSDT_1h-m2.csv')

        # columns setting
        self.dataframes = {column:  self.dataframe[column] for column in self.dataframe.columns}

    def save_csv(self, dataframe:pd.DataFrame, path:str):
        dataframe.to_csv(path, index=False, float_format='%.2f')

if __name__ == '__main__':
    manager = CSVManager()
    manager.save_csv(manager.dataframe,'./assets/Binance_BTCUSDT_1h-m3.csv')

1518051600