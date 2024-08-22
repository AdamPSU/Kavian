import pandas as pd
import numpy as np

from kavian.eda.config import FLOAT, NUM, DTYPE_PRIORITY

def _process_mode(dataframe: pd.DataFrame):
    """
    Should test this function
    """

    modes, percents = [], []
    size = dataframe.value_counts().sum()

    for col in dataframe:
        mode = col.mode()
        mode_size = col.value_counts()[0]

        if mode in FLOAT:
            mode = np.round(mode, 5)

        percent = f'{mode_size / size * 100:.2f}%'

        modes.append(mode); percents.append(percent)

    return modes, percents


def _process_num(dataframe: pd.DataFrame):
    min_values, max_values = [], []

    for col in dataframe:
        if dataframe[col].dtype.name not in NUM:
            min_values.append(' '); max_values.append(' ')
            continue

        min_value = f'{dataframe[col].min():.3f}'
        max_value = f'{dataframe[col].max():.3f}'

        min_values.append(min_value); max_values.append(max_value)

    return min_values, max_values


def _process_memory(dataframe: pd.DataFrame):
    memory = dataframe.memory_usage(deep=True).sum()
    kb = 1024

    if memory < kb:
        return f'{memory} bytes'
    elif memory < kb ** 2:
        return f'{memory / kb:.2f} KB'
    elif memory < kb ** 3:
        return f'{memory / kb ** 2:.2f} MB'
    else:
        return f'{memory / kb ** 3:.2f} GB'


def info(dataframe: pd.DataFrame, categorical='auto'):
    features = sorted(dataframe.columns, key=lambda col: DTYPE_PRIORITY[dataframe[col].dtype.name], )
    dataframe = dataframe[features]

    null = dataframe.isna().sum()
    null_percents = null / len(dataframe) * 100
    null_percents = null_percents.apply(lambda x: f'{x:.2f}')

    most_common, most_common_percents = _process_mode(dataframe)
    min_values, max_values = _process_num(dataframe)

    unique = dataframe.nunique()
    dtypes = dataframe.dtypes

    analysis = pd.DataFrame({'Dtype': dtypes,
                                  'Unique': unique,
                                  'Null': null,
                                  'Null %': null_percents,
                                  'Most Common': most_common,
                                  'Most Common %': most_common_percents,
                                  'Min': min_values,
                                  'Max': max_values},
                            index=features)

    analysis = analysis.style.set_table_styles([
        {'selector': 'td, th', 'props': [('border', '0.2px solid white')]},
    ])

    memory = _process_memory(dataframe)
    num_cols = len(dataframe.columns)

    print(f'no. columns: {num_cols} • table size: {len(dataframe)} • ' +
          f'memory usage: {memory}')

    return analysis




















