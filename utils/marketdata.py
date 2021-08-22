"""마켓(binance)에서 데이터를 얻어오는 함수를 구현.
"""


import time
import datetime
import math
import requests
import numpy as np
import pandas as pd



def get_market_data(start_time, end_time, symbol, interval):
    """startTime부터 endTime까지 interval간격의 모든 데이터를 반환하는 함수

    Parameters
    ----------
        start_time : String
            시작시간  %Y-%m-%d %H:%M:%S 형식 문자열
        end_time : String
            끝 시간 %Y-%m-%d %H:%M:%S 형식 문자열
        symbol : String
            마켓 이름 ex) 'BTCUSDT'
        interval : String
            검색간격 ex) '1d'
    Returns
    -------
    dataframe : 2-D DataFrame
        limit X 5 크기의 2차원 pandas.dataframe.
    """
    start_time_unix = time_to_unixtime(start_time)
    end_time_unix= time_to_unixtime(end_time)
    interval_time = interval_to_deltatime(interval)

    # ? parameters 검증
    try:
        if (end_time_unix - start_time_unix) < 0:
            raise MinusTimeError()
        if (end_time_unix - start_time_unix) < interval_time:
            raise MinimalDeltaTimeError()

    except MinusTimeError:
        print('종료 시간은 시작시간보다 나중이어야 합니다.')
    except MinimalDeltaTimeError:
        print('종료 시간과 시작시간 간의 간격은 inteval의 크기이상이어야 합니다. ')

    # ? 실행
    num = (end_time_unix - start_time_unix)/interval_time  # 구해야할 데이터의 크기 즉, 행의 길이
    if num <= 1000:
        # print('symbol=', symbol, ' startTime=', startTime,
        #       ' interval=', interval, ' limit=', int(num))
        data = get_history(symbol=symbol, start_time=start_time,
                           interval=interval, limit=int(num))
    else:
        result = np.zeros((0, 5))
        repeat = math.ceil(num/1000)  # 반복문 돌리는 횟수
        remain = int(num % 1000)  # 나머지. 마지막 반복에서 사용.

        for i in range(repeat):
            if i == repeat-1:  # 마지막 반복일때
                history = get_history_as_unixtime(
                    symbol=symbol, start_time=start_time_unix + i * interval_time * 1000+(1 if i > 0 else 0), interval=interval, limit=remain)
                result = np.concatenate((result, history), axis=0)
            else:
                history = get_history_as_unixtime(
                    symbol=symbol, start_time=start_time_unix + i * interval_time * 1000+(1 if i > 0 else 0), interval=interval, limit=1000)
                result = np.concatenate((result, history), axis=0)
        data = result

    # ? make dataframe
    index = time_index(start_time, end_time, interval)
    print(num)
    print(data.shape)
    print(index.shape)
    
    dataframe = pd.DataFrame(data, columns=[
        'open', 'high', 'low', 'close', 'volume'],index=index[:-1])
    dataframe = dataframe.astype('float')
    return dataframe


def time_to_unixtime(str):
    """
    Parameters
    ----------
    str : String
        %Y-%m-%d %H:%M:%S 형식 문자열


    Returns
    -------
    int : Int
        입력값을 Unix 시간으로 바꾼 값
    """
    return int(time.mktime(datetime.datetime.strptime(
        str, '%Y-%m-%d %H:%M:%S').timetuple())) * 1000


def interval_to_deltatime(interval: str):
    """
    #! 유연한 처리가 가능하도록 수정필요.
    """
    result = 0
    if interval.__contains__('d') | interval.__contains__('D'):
        result = 86400 * int(interval.replace('d', ' ').replace('D', ' '))

    elif interval.__contains__('h') | interval.__contains__('H'):
        result = 3600 * int(interval.replace('h', ' ').replace('H', ' '))

    elif interval.__contains__('m') | interval.__contains__('M'):
        result = 60 * int(interval.replace('m', ' ').replace('M', ' '))

    elif interval.__contains__('s') | interval.__contains__('S'):
        result = 1 * int(interval.replace('s', ' ').replace('S', ' '))

    return result * 1000


def get_history(symbol, start_time, interval, limit=1):
    """
    Parameters
    ----------
    symbol : String
        마켓 이름 ex) 'BTCUSDT'
    start_time : String
        시작시간  %Y-%m-%d %H:%M:%S 형식 문자열
    interval : String
        검색간격 ex) '1d'
    limit : Int
        최대 검색 횟수 max=1000

    Returns
    -------
    np.array : 2-D Array
        limit X 5 크기의 2차원 np행렬 [open,high,low,close,volume] 순서

    """
    res = requests.get('https://api.binance.com/api/v3/klines', params={
        'symbol': symbol, 'interval': interval, 'startTime': time_to_unixtime(start_time), 'limit': limit})
    data = np.array(res.json())[:, 1:6]  # [open, high ,low ,close ,volume]
    return data


def get_history_as_unixtime(symbol, start_time, interval, limit):
    """
    Parameters
    ----------
    symbol : String
        마켓 이름
    start_time : Long
        시작시간  unix time 형식
    interval : String
        검색간격
    limit : Int
        최대 검색 횟수 max=1000

    Returns
    -------
    np.array : 2-D Array
        limit X 5 크기의 2차원 np행렬 [open,high,low,close,volume] 순서

    """
    res = requests.get('https://api.binance.com/api/v3/klines', params={
        'symbol': symbol, 'interval': interval, 'start_time': start_time, 'limit': limit})
    data = np.array(res.json())[:, 1:6]  # [open, high ,low ,close ,volume]
    return data

def time_index(start_time:str, end_time:str, interval:str):
    """데이터프레임의 index 부분 Generator

    Parameters
    ----------
    start_time : str 
        e.g. 2018-01-01 00:00:00
    end_time : str
        [description]
    interval : str 
        e.g. 4h
    
    Returns
    -------
    TimeDeltaIndex
    """

    def interval_checker(interval: str):
        """pd.timedelta_range의 freq 인자에 맞게 interval의 형식을 수정.

        Parameters
        ----------
        interval : str
            like '4h', '30m', '1d' ...

        Returns
        -------
        str
            '4h', '30T', '1d' ...
        """
        if interval.__contains__('m'):
            interval = interval.replace('m','T')
        return interval
    
    start_datetime = datetime.datetime.fromisoformat(start_time)
    end_datetime = datetime.datetime.fromisoformat(end_time)

    index = pd.timedelta_range(start='0 days', end=end_datetime-start_datetime,freq=interval_checker(interval))
    index = index.__add__(start_datetime)

    return index

class MinusTimeError(Exception):
    pass


class MinimalDeltaTimeError(Exception):
    pass
