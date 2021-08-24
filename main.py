"""
"""

import torch
import datasets
from torchvision.transforms import ToTensor

start_time = '2021-07-14 00:00:00'
end_time = '2021-08-16 00:00:00'
symbol = 'BTCUSDT'
interval = '1m'
settings = {'start':start_time, 'end':end_time, 'symbol':symbol,'interval':interval, 'past_length':10, 'future_length':5}

if __name__ == '__main__':

    dataset = datasets.PastFutureDataset(**settings)
    # network = NETWORK
    # network setting()
    # training()
    # testing()






