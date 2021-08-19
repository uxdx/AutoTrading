"""요청에 따라 적절한 데이터셋을 반환하는API를 구현
"""

import sys
from data.datasetmaker import PastFutureDataMaker
from data.marketdata import get_market_data


if __name__ == '__main__':
    start_time = '2021-07-14 00:00:00'
    end_time = '2021-07-16 00:00:00'
    dataframe = get_market_data(
        start_time=start_time, end_time=end_time, symbol='BTCUSDT', interval='1h')

    for arg in sys.argv:
        print(arg)

    if sys.argv[1] == 'create':
        """데이터셋을 생성.
        argv[2] : 경로
        """
