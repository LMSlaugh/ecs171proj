import pandas

def read_data():
    data_csv = pandas.read_csv('scc_data_to_use.csv', index_col = 0, parse_dates=True, infer_datetime_format=True)
    return data_csv
