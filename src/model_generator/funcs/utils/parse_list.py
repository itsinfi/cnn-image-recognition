from json import loads, JSONDecodeError
from pandas import isna

def parse_list(val):
    if isna(val) or val == '':
        return []
    try:
        return loads(val)
    except JSONDecodeError:
        return []