import ccxt
import requests
import time
import datetime
import numpy as np
from custom_functions import get_market_data, get_history

binance = ccxt.binance()
ticker = binance.fetch_ticker('ETH/USDT')
binance = ccxt.binance({
    'apiKey': '0pf5ROmfYGNAI1JRfNmVTzuBPlSkKdsX4TYCnJdL4Z6qqOgjH7uDaSFNPtmiPAZb',
    # ! 다시 볼 수 없으니 소중히 보관
    'secret': 'pkUerOVtdHz1GMmWTfoEYpRijliBFYblRyrZxZDNn2h5bn3RPDMQ5zKH7jrXO6Cl',
})
markets = binance.load_markets()
balance = binance.fetch_balance()
# print(balance['USDT']['free'], balance['USDT']
#       ['used'], balance['USDT']['total'])
# print(ticker['high'])
# print(ticker['low'])
# print(ticker['open'])
# print(ticker['close'])


# print(start)

# history = get_history(
#     symbol='BTCUSDT', startTime=startTime, interval='1d', limit=3)
# print(history.shape)

# print(history)
# print(type(req1.json()))


def test_get_market_data():
    startTime = '2021-07-14 00:00:00'
    endTime = '2021-07-16 00:00:00'
    history = get_market_data(
        startTime=startTime, endTime=endTime, symbol='BTCUSDT', interval='1h')

    if history.shape == np.zeros((48, 5)).shape:
        print('test_get_market_data Clear')


if __name__ == '__main__':
    test_get_market_data()
