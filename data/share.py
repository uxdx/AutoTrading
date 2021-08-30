def make_file_name(isData:bool,algorithm_name,tags,start,end,interval):
    return ''.join(['Data_'if isData else 'Targets_', algorithm_name, tags,'_',start,'~',end,'_',interval])
def default_data_path():
    return './assets/'

def datetime_to_unixtime(date_time:str) -> int:
    """Convert datetime('%Y-%m-%d %H:%M:%S') to unixtime"""
    import datetime
    return int(datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S').timestamp())

def to_thousands(num):
    return num * 1000

def none_init(dict, key):
    """딕셔너리와 그 키값을 인수로 받아,
    해당하는 딕셔너리에 해당하는 키값이나 키 자체가 없다면 None을 반환함.
    """
    try:
        return None if dict[key] is None else dict[key]
    except KeyError:
        return None