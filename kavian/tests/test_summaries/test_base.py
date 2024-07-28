import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from kavian.summary._base import (
    BaseSummary,
    BaseRegressorSummary,
    BaseClassifierSummary
)

X_RANDOM = np.random.rand(100, 3)
Y_RANDOM = np.random.rand(100)

def test_base_not_instance(get_diabetes):
    # The BaseSummary class is not meant to be an instance

    linear_regr, X, y = get_diabetes

    with pytest.raises(TypeError):
        BaseSummary(linear_regr, X, y)


def test_compatibility(get_diabetes, get_breast_cancer):
    # Both Numpy arrays and Pandas Dataframes should be compatible

    try:
        linear_regr, numpy_X, numpy_y = get_diabetes
        BaseRegressorSummary(linear_regr, numpy_X, numpy_y)

        pandas_X, pandas_y = pd.DataFrame(numpy_X), pd.Series(numpy_y)
        BaseRegressorSummary(linear_regr, pandas_X, pandas_y)

        logistic_classifier, numpy_X, numpy_y = get_breast_cancer
        BaseClassifierSummary(logistic_classifier, numpy_X, numpy_y)

        pandas_X, pandas_y = pd.DataFrame(numpy_X), pd.Series(numpy_y)
        BaseClassifierSummary(logistic_classifier, pandas_X, pandas_y)
    except Exception as e:
        pytest.fail(f"Compatibility Error: {e}")


def test_model_fitted(get_diabetes):
    # Model should be fitted prior to summary

    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        BaseRegressorSummary(LinearRegression(), X_RANDOM, Y_RANDOM)

    with pytest.raises(NotFittedError):
        BaseClassifierSummary(LogisticRegression(), X_RANDOM, Y_RANDOM)


def test_regressor_statistics(get_diabetes, get_california_housing):
    """
    Tests accuracy of regression summary statistics.

    Hardcoded values are taken from statsmodels.api's OLS regression
    model fitted on the same data.
    """

    linear_diabetes, X_diabetes, y_diabetes = get_diabetes
    linear_calif, X_calif, y_calif = get_california_housing

    diabetes = BaseRegressorSummary(linear_diabetes, X_diabetes, y_diabetes)
    california = BaseRegressorSummary(linear_calif, X_calif, y_calif)

    assert np.ceil(diabetes.get_rss()) == 1_263_986
    assert np.ceil(california.get_rss()) == 10_822

    assert np.ceil(diabetes.get_log_likelihood()) == -2_385
    assert np.ceil(california.get_log_likelihood()) == -22_623

    assert np.ceil(diabetes.get_aic()) == 4_794
    assert np.ceil(california.get_aic()) == 45_266

    assert np.ceil((diabetes.get_bic())) == 4_839
    assert np.ceil((california.get_bic())) == 45_337

    diabetes_r2, diabetes_adj_r2 = np.round(diabetes.get_r2(), 3)
    calif_r2, calif_adj_r2 = np.round(california.get_r2(), 3)

    assert diabetes_r2 == 0.518
    assert diabetes_adj_r2 == 0.507

    assert calif_r2 == 0.606
    assert calif_adj_r2 == 0.606

















