import pytest

from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_breast_cancer
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

@pytest.fixture
def get_diabetes():
    """
    Load and return a fitted Linear Regression model on
    the Diabetes toy dataset (Regression)

    :returns

    - estimator : fitted model on Diabetes dataset
    - X : data matrix
    - y : response vector
    """

    X, y = load_diabetes(return_X_y=True)

    estimator = LinearRegression()
    estimator.fit(X, y)

    return estimator, X, y


@pytest.fixture
def get_california_housing():
    """
    Load and return a fitted Linear Regression model on
    the California Housing dataset (Regression)

    :returns

    - estimator : fitted model on California Housing dataset
    - X : data matrix
    - y : response vector
    """

    X, y = fetch_california_housing(return_X_y=True)

    estimator = LinearRegression()
    estimator.fit(X, y)

    return estimator, X, y

@pytest.fixture
def get_breast_cancer():
    """
    Load and return a fitted Logistic Regression model on
    the Breast Cancer Wisconsin dataset (Classification)

    :returns

    - estimator : fitted Logistic Regression model
    - X : data matrix
    - y : response vector
    """

    X, y = load_breast_cancer(return_X_y=True)

    estimator = LogisticRegression()
    estimator.fit(X, y)

    return estimator, X, y
