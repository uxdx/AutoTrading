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