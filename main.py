"""
파라미터로 범위와 간격을 지정하면 
데이터크기x5 의 행열을 반환하는 api 구현
"""



if __name__ == '__main__':
    import pandas as pd
    import datetime
    start_time = '2018-01-01 00:00:00'
    end_time = '2018-01-01 23:59:59'


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


    index = time_index(start_time=start_time, end_time=end_time, interval='30m')
    print(index)




    # start_datetime = datetime.datetime.combine(start_date, start_time)
    # start_datetime = datetime.datetime.fromisoformat(start_time)
    # end_datetime = datetime.datetime.fromisoformat(end_time)

    # delta = pd.timedelta_range(start='0 days', end=end_datetime-start_datetime,freq='4h')
    # delta = delta.__add__(start_datetime)
    # print(delta)

    # print(start_datetime)
