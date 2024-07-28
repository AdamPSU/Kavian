import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from kavian.tables.base import (
    BaseSummary,
    SimpleRegressorSummary,
    SimpleClassifierSummary
)

X_RANDOM = np.random.rand(100, 3)
Y_RANDOM = np.random.rand(100)


def test_compatibility(get_diabetes, get_breast_cancer):
    # Both Numpy arrays and Pandas Dataframes should be compatible

    try:
        linear_regr, numpy_X, numpy_y = get_diabetes
        SimpleRegressorSummary(linear_regr, numpy_X, numpy_y)

        pandas_X, pandas_y = pd.DataFrame(numpy_X), pd.Series(numpy_y)
        SimpleRegressorSummary(linear_regr, pandas_X, pandas_y)

        logistic_classifier, numpy_X, numpy_y = get_breast_cancer
        SimpleClassifierSummary(logistic_classifier, numpy_X, numpy_y)

        pandas_X, pandas_y = pd.DataFrame(numpy_X), pd.Series(numpy_y)
        SimpleClassifierSummary(logistic_classifier, pandas_X, pandas_y)
    except Exception as e:
        pytest.fail(f"Compatibility Error: {e}")


def test_regressor_statistics(get_diabetes, get_california_housing):
    """
    Tests accuracy of regression tables statistics.

    Hardcoded values are taken from statsmodels.api's OLS regression
    model fitted on the same data.
    """

    linear_diabetes, X_diabetes, y_diabetes = get_diabetes
    linear_calif, X_calif, y_calif = get_california_housing

    diabetes = SimpleRegressorSummary(linear_diabetes, X_diabetes, y_diabetes)
    california = SimpleRegressorSummary(linear_calif, X_calif, y_calif)

    assert np.ceil(diabetes.get_rss()) == 1_263_986
    assert np.ceil(california.get_rss()) == 10_822

    assert np.ceil(diabetes.get_log_likelihood()) == -2_385
    assert np.ceil(california.get_log_likelihood()) == -22_623

    assert np.ceil(diabetes.get_aic()) == 4_794
    assert np.ceil(california.get_aic()) == 45_266

    assert np.ceil((diabetes.get_bic())) == 4_839
    assert np.ceil((california.get_bic())) == 45_337

    assert np.round(diabetes.get_r2(), 3) == 0.518
    assert np.round(diabetes.get_adj_r2(), 3) == 0.507

    assert np.round(california.get_r2(), 3) == 0.606
    assert np.round(california.get_adj_r2(), 3) == 0.606

















