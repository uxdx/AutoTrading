def make_file_name(isData:bool,algorithm_name,tags,start,end,interval):
    return ''.join(['Data_'if isData else 'Targets_', algorithm_name, tags,'_',start,'~',end,'_',interval])
def default_data_path():
    return './assets/'