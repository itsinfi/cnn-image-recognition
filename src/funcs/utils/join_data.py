from pandas import DataFrame, merge

def join_data(csv_files: list[DataFrame]) -> DataFrame:
    # join_1 = merge(left=csv_files[3], right=csv_files[2], on=['itemID', 'userID'])
    # join_2 = merge(left=csv_files[2], right=csv_files[1], on='itemID')
    # joined_csv = merge(left=csv_files[0], right=join_2, left_on='category', right_on='categories')
    # return joined_csv
    return csv_files[0]