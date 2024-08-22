import pandas as pd

def info(dataframe: pd.DataFrame, categorical='auto'):
    """
    include: missing counts, unique category counts,
    """

    features = dataframe.columns.tolist()

    missing = dataframe.isna().sum()
    unique = dataframe.nunique()
    dtypes = dataframe.dtypes

    analysis = pd.DataFrame({'Dtype': dtypes, 'Missing': missing, 'Unique': unique},
                            index=features)
    analysis = analysis.style.set_table_styles([
        {'selector': 'td, th', 'props': [('border', '0.2px solid white')]},
    ])

    return analysis




















