import pandas as pd
import numpy as np

from kavian.eda.config import FLOAT, NUM, CAT, DTYPE_PRIORITY
from kavian import KavianError


def _process_mode(dataframe: pd.DataFrame):
    """
    Should test this function
    """

    modes, percents = [], []
    size = len(dataframe)

    for col in dataframe:
        mode = dataframe[col].mode().iloc[0]
        mode_size = dataframe[col].value_counts().iloc[0]

        if dataframe[col].dtype.name in FLOAT:
            mode = f'{mode:.3f}'

        percent = f'{mode_size / size * 100:.2f}%'

        modes.append(mode);
        percents.append(percent)

    return modes, percents


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


def info(dataframe: pd.DataFrame, numerical=True, categorical=True):
    if not categorical and not numerical:
        raise KavianError(
            "Neither categorical nor numerical features were supplied. Please include at least "
            "one parameter for exploratory analysis."
        )

    if numerical:
        num = dataframe.select_dtypes(include=NUM)
        dataframe = num

    if categorical:
        cat = dataframe.select_dtypes(include=CAT)
        dataframe = cat

    sorted_features = sorted(dataframe.columns, key=lambda col: DTYPE_PRIORITY[dataframe[col].dtype.name])
    dataframe = dataframe[sorted_features]

    null = dataframe.isna().sum()
    null_percents = null / len(dataframe) * 100
    null_percents = null_percents.apply(lambda x: f'{x:.2f}%')

    most_common, most_common_percents = _process_mode(dataframe)

    unique = dataframe.nunique()
    dtypes = dataframe.dtypes

    data = {'Dtype': dtypes,
            'Unique': unique,
            'Null': null,
            'Null %': null_percents,
            'Most Common': most_common,
            'Most Common %': most_common_percents}

    analysis = pd.DataFrame(data, index=sorted_features)
    # Color white
    analysis = analysis.style.set_table_styles([
        {'selector': 'td, th', 'props': [('border', '0.2px solid white')]},
    ])

    memory = _process_memory(dataframe)
    num_cols = len(dataframe.columns)

    print(f'table size: {len(dataframe)} • no. columns: {num_cols} • memory usage: {memory}')

    return analysis


def describe(dataframe: pd.DataFrame, numerical=True):
    categorical = None

    if not categorical and not numerical:
        raise KavianError(
            "Neither categorical nor numerical features were supplied. Please include "
            "one parameter for exploratory analysis."
        )

    if numerical:
        dataframe = dataframe.select_dtypes(include=NUM)
        sorted_features = sorted(dataframe.columns, key=lambda col: DTYPE_PRIORITY[dataframe[col].dtype.name])

        dataframe = dataframe[sorted_features]

        analysis = pd.DataFrame({
            'Count': dataframe.count(),
            'Mean': dataframe.mean(),
            'Stdev': dataframe.std(),
            'Min': dataframe.min(),
            '25%': dataframe.quantile(0.25),
            '50%': dataframe.median(),
            '75%': dataframe.quantile(0.75),
            'Max': dataframe.max(),
            'Skewness': dataframe.skew()
        }, index=sorted_features)

        analysis = analysis.applymap(lambda x: f'{x:.3f}' if isinstance(x, float) else str(x))

    analysis = analysis.style.set_table_styles([
        {'selector': 'td, th', 'props': [('border', '0.2px solid white')]},
    ])

    return analysis






