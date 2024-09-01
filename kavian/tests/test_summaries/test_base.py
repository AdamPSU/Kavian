import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression
from rich.panel import Panel

from kavian.tables.base import (
    BaseRegressorSummary,
    SimpleRegressorSummary,
    SimpleClassifierSummary
)

X_RANDOM = np.random.rand(100, 3)
Y_RANDOM = np.random.rand(100)

class TooManyEntriesSummary(BaseRegressorSummary):
    """Edge case with just too many entries."""

    def make_entries(self):
        empty = ("dummy: ", "value")
        available_space = 5

        return [empty]*(available_space + 1)


    def summary(self):
        model_entries = self.make_entries()
        model_table = self.create_table(*model_entries)

        self.console.print(Panel(model_table))


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


def test_too_many_entries():
    with (pytest.raises(ValueError)):
        model = LinearRegression()
        model.fit(X_RANDOM, Y_RANDOM)

        test = TooManyEntriesSummary(model, X_RANDOM, Y_RANDOM)
        test.summary()




















