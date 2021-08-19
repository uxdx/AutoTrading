# Get Data From Binance

binance api를 통해 ML에 사용할 데이터를 확보, 가공하는 

data/ : 머신러닝에 필요한 데이터를 구성하고 저장하고, 가져오는 기능을 구현한 모듈이 보관되어있음
  getmarketdata.py : binance api 등에서 마켓 자료를 가져오는 모듈
  datasetmaker.py : 마켓 데이터를 머신러닝에 쓸 수 있는 형태의 데이터셋으로 만드는 모듈
  
# Usage

1. Get binance market data 

        from data.datasetmaker import PastFutureDataMaker
        from data.marketdata import get_market_data

        start_time = '2021-07-14 00:00:00'
        end_time = '2021-07-16 00:00:00'
        dataframe = get_market_data(
            start_time=start_time, end_time=end_time, symbol='BTCUSDT', interval='1h')

2. Make DataSet

        maker = PastFutureDataMaker(dataframe,26,9)

        dataset = maker.make_bundle()
        
3. Save DataSet

        dataset.save('./temp/', 'ds1')
        
4. Load DataSet

        dataset = DataSet().load('./temp/', 'ds1')

        x, y = dataset.x, dataset.y

        print(dataset)
