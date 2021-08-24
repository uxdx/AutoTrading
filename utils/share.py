def make_file_name(isData:bool,algorithm_name,tags,start,end,interval):
    return ''.join(['Data_'if isData else 'Targets_', algorithm_name, tags,'_',start,'~',end,'_',interval])
def default_data_path():
    return './assets/'

def datetime_to_unixtime(date_time:str) -> int:
    """Convert datetime('%Y-%m-%d %H:%M:%S') to unixtime"""
    import datetime
    return int(datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S').timestamp())

def enum_to_unixtime(interval:str) -> int:
    result = 0
    if interval.__contains__('d') | interval.__contains__('D'):
        result = 86400 * int(interval.replace('d', ' ').replace('D', ' '))

    elif interval.__contains__('h') | interval.__contains__('H'):
        result = 3600 * int(interval.replace('h', ' ').replace('H', ' '))

    elif interval.__contains__('m') | interval.__contains__('M'):
        result = 60 * int(interval.replace('m', ' ').replace('M', ' '))

    elif interval.__contains__('s') | interval.__contains__('S'):
        result = 1 * int(interval.replace('s', ' ').replace('S', ' '))
    return result