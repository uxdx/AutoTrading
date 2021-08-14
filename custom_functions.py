import time
import datetime
import numpy as np
import requests
import math
import exceptions


def time_to_unixTime(str):
    """
    params:
        str: %Y-%m-%d %H:%M:%S 형식 문자열
    returns:
        int: 입력값을 Unix 시간으로 바꾼 값
    """
    return int(time.mktime(datetime.datetime.strptime(
        str, '%Y-%m-%d %H:%M:%S').timetuple())) * 1000


def interval_to_deltaTime(interval):
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


def get_history(symbol='BTCUSDT', startTime='2021-08-14 00:00:00', interval='1d', limit=1):
    """
    params:
        symbol: 마켓 이름
        startTime: 시작시간  %Y-%m-%d %H:%M:%S 형식 문자열
        interval: 검색간격
        limit: 최대 검색 횟수 max=1000 
    returns:
        np.array: limit X 5 크기의 2차원 np행렬 [open,high,low,close,volume] 순서

    """
    res = requests.get('https://api.binance.com/api/v3/klines', params={
        'symbol': symbol, 'interval': interval, 'startTime': time_to_unixTime(startTime), 'limit': limit})
    print(res)
    data = np.array(res.json())[:, 1:6]  # [open, high ,low ,close ,volume]
    return data


def get_history_as_unixTime(symbol, startTime, interval, limit):
    """
    params:
        symbol(str): 마켓 이름
        startTime(long): 시작시간  unix time 형식
        interval(str): 검색간격
        limit(int): 최대 검색 횟수 max=1000 
    returns:
        np.array: limit X 5 크기의 2차원 np행렬 [open,high,low,close,volume] 순서

    """
    res = requests.get('https://api.binance.com/api/v3/klines', params={
        'symbol': symbol, 'interval': interval, 'startTime': startTime, 'limit': limit})
    print(res)
    data = np.array(res.json())[:, 1:6]  # [open, high ,low ,close ,volume]
    return data


def get_market_data(startTime, endTime, symbol='BTCUSDT', interval='1d'):
    """
    startTime부터 endTime까지 interval간격의 모든 데이터를 반환하는 함수
    params:
        startTime(str): 시작시간  %Y-%m-%d %H:%M:%S 형식 문자열
        endTime(str): 끝 시간 %Y-%m-%d %H:%M:%S 형식 문자열
        symbol(str): 마켓 이름
        interval(str): 검색간격
    returns:
        np.array: limit X 5 크기의 2차원 np행렬 [open,high,low,close,volume] 순서
    """
    start_time = time_to_unixTime(startTime)
    end_time = time_to_unixTime(endTime)
    interval_time = interval_to_deltaTime(interval)

    # endTime - startTime이 0이상인지, 1 interval 이상인지 체크
    try:
        if (end_time - start_time) < 0:
            raise MinusTimeError()
        if (end_time - start_time) < interval_time:
            raise MinimalDeltaTimeError()

    except MinusTimeError:
        print('종료 시간은 시작시간보다 나중이어야 합니다.')
    except MinimalDeltaTimeError:
        print('종료 시간과 시작시간 간의 간격은 inteval의 크기이상이어야 합니다. ')

    # (endTime - startTime) / interval 이 1000을 넘지 않는 지 체크
    num = (end_time - start_time)/interval_time  # 구해야할 데이터의 크기 즉, 열의 크기
    print('num: ', num)
    if num <= 1000:
        # print('symbol=', symbol, ' startTime=', startTime,
        #       ' interval=', interval, ' limit=', int(num))
        return get_history(symbol=symbol, startTime=startTime, interval=interval, limit=int(num))
    else:
        result = np.zeros((0, 5))
        epoch = math.ceil(num/1000)  # 반복문 돌리는 횟수
        remain = int(num % 1000)  # 나머지. 마지막 반복에서 사용.

        for i in range(epoch):
            if i == epoch-1:  # 마지막 반복일때
                history = get_history_as_unixTime(
                    symbol=symbol, startTime=start_time + i * interval_time * 1000+(1 if i > 0 else 0), interval=interval, limit=remain)
                result = np.concatenate((result, history), axis=0)
            else:
                history = get_history_as_unixTime(
                    symbol=symbol, startTime=start_time + i * interval_time * 1000+(1 if i > 0 else 0), interval=interval, limit=1000)
                result = np.concatenate((result, history), axis=0)
        return result
