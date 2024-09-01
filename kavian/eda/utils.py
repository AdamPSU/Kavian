import pandas as pd

from kavian import KavianError
from kavian.eda.config import NUM, CAT


def _subset_handler(dataframe, subset):
    if subset is None:
        return dataframe

    if isinstance(subset, list) or isinstance(subset, pd.Index):
        subset = subset
    elif subset == 'numerical':
        subset = dataframe.select_dtypes(include=NUM).columns
    elif subset == 'categorical':
        subset = dataframe.select_dtypes(include=CAT).columns
    else:
        raise KavianError(
            "Subset parameter must be set to one of 'numerical', 'categorical', " +
            f"or a list or pandas index object containing desired columns. Got: {subset} instead."
        )

    dataframe = dataframe[subset]

    return dataframe