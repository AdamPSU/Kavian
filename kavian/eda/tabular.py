import pandas as pd

from kavian.eda.config import FLOAT, NUM, CAT, DTYPE_PRIORITY
from kavian.eda.utils import subset_handler
from kavian.eda._colors import color_thresholded_column, color_adversarial_column, color_outliers

def _process_mode(dataframe: pd.DataFrame):
    """
    Returns the mode and mode percentage of all
    specified dataframe columns.
    """

    modes, percents = [], []
    size = len(dataframe)

    for col in dataframe:
        mode = dataframe[col].mode().iloc[0]
        mode_size = dataframe[col].value_counts().iloc[0]

        if dataframe[col].dtype.name in FLOAT:
            mode = f'{mode:.3f}'

        percent = mode_size / size * 100

        modes.append(mode); percents.append(percent)

    return modes, percents


def _process_memory(dataframe: pd.DataFrame):
    """
    Provides memory usage information for a given dataframe.
    """

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


def info(dataframe: pd.DataFrame, colored_output=True, subset=None):
    dataframe = subset_handler(dataframe, subset)

    # Sort features
    features = sorted(dataframe.columns, key=lambda col: DTYPE_PRIORITY[dataframe[col].dtype.name])
    dataframe = dataframe[features]

    null = dataframe.isna().sum()
    null_percents = null / len(dataframe) * 100

    most_common, most_common_percents = _process_mode(dataframe)

    unique = dataframe.nunique()
    dtypes = dataframe.dtypes

    data = {'Dtype': dtypes,
            'Unique': unique,
            'Null': null,
            'Null %': null_percents,
            'Top': most_common,
            'Top %': most_common_percents}

    analysis = pd.DataFrame(data, index=features).style
    analysis = analysis.format({'Null %': '{:.2f}%', 'Top %': '{:.2f}%'})

    if colored_output:
        analysis = analysis.map(lambda x: color_thresholded_column(x, low_threshold=3, high_threshold=10),
                                subset=['Null %'])
        analysis = analysis.map(lambda x: color_thresholded_column(x, low_threshold=20, high_threshold=40),
                                subset=['Top %'])

    analysis = analysis.set_table_styles([
        {'selector': 'td, th', 'props': [('border', '0.2px solid white')]},
    ])

    memory = _process_memory(dataframe)
    num_cols = len(dataframe.columns)
    num_obs = len(dataframe)

    print(f'table size: {num_obs} • no. columns: {num_cols} • memory usage: {memory}')

    return analysis


def describe(dataframe: pd.DataFrame, categorical=False, colored_output=True):
    if not categorical:
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

        if colored_output:
            analysis = color_outliers(analysis, cols=['Min', 'Max'])
            analysis = analysis.map(lambda x: color_adversarial_column(x, threshold=0),
                                    subset=['Skewness'])

        if hasattr(analysis, 'style'):
            analysis = analysis.style

        analysis = analysis.format('{:.3f}')

    else:
        dataframe = dataframe.select_dtypes(include=CAT)
        sorted_features = sorted(dataframe.columns, key=lambda col: DTYPE_PRIORITY[dataframe[col].dtype.name])

        top, top_percent = _process_mode(dataframe)

        analysis = pd.DataFrame({
            'Count': dataframe.count(),
            'Unique': dataframe.nunique(),
            'Top': top,
            'Top %': top_percent

        }, index=sorted_features)

        analysis = analysis.style

        if colored_output:
            analysis = analysis.map(
                lambda x: color_thresholded_column(x, low_threshold=20, high_threshold=40),
                subset=['Top %'])

        analysis = analysis.format({'Top %': '{:.2f}%'})

    analysis.set_table_styles([
        {'selector': 'td, th', 'props': [('border', '0.2px solid white')]},
    ])

    return analysis
