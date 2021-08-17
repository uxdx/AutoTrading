"""마켓(binance)에서 데이터를 얻어오는 함수를 구현.
"""


import time
import datetime
import math
import requests
import numpy as np
import pandas as pd
import exceptions


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


def interval_to_deltatime(interval):
    """
    #! 유연한 처리가 가능하도록 수정필요.
    """
    result = 0
    if interval == '1d':
        result = 86400
    elif interval == '4h':
        result = 14400
    elif interval == '1h':
        result = 3600
    elif interval == '1m':
        result = 60
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
            raise exceptions.MinusTimeError()
        if (end_time_unix - start_time_unix) < interval_time:
            raise exceptions.MinimalDeltaTimeError()

    except exceptions.MinusTimeError:
        print('종료 시간은 시작시간보다 나중이어야 합니다.')
    except exceptions.MinimalDeltaTimeError:
        print('종료 시간과 시작시간 간의 간격은 inteval의 크기이상이어야 합니다. ')

    # ? 실행
    num = (end_time_unix - start_time_unix)/interval_time  # 구해야할 데이터의 크기 즉, 열의 크기
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
    dataframe = pd.DataFrame(data, columns=[
        'open', 'high', 'low', 'close', 'volume'])
    dataframe = dataframe.astype('float')
    return dataframe