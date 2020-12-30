from functools import reduce

import pandas_profiling
from pandas import read_csv, read_table, merge, concat


def fn_to_df_(filename, from_='raw_datasets', samples=0, describe=False):
    fn = f'{from_}/{filename}'

    if '.csv' in filename:
        df = read_csv(fn)

    elif '.xyz' in filename:
        df = read_table(fn, skiprows=5, sep='\s+', dtype=float,
                        names=['X', 'Y', 'Z', filename.split('.')[0]])

    if samples:
        df = df.sample(samples)

    if describe:
        dfd = df.describe()
        dfd.insert(0, 'Stat', dfd.index)
        return df, dfd

    return df


def gen_profile_from_df(df, filename):
    pf = pandas_profiling.ProfileReport(df)
    pf.to_file(f'templates/{filename}.html')


def simple_merge(dfs, on_, how_):
    return reduce(lambda left, right: merge(left, right, on=on_, how=how_), dfs)


def simple_concat(dfs):
    return concat(dfs)


def get_col_manes(ds_name):
    dataset = read_csv(f'datasets/{ds_name}.csv', nrows=1)
    return list(dataset.columns)


def save_to_(df, name):
    df.to_csv(f'predictions/{name}.csv', index=False)
    return True
