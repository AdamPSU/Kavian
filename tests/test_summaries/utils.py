from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing
)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def get_diabetes():
    """
    Load and return the train and target variables for the
    Diabetes dataset (regression).
    """

    return load_diabetes(return_X_y=True)


def get_california_housing():
    """
    Load and return the train and target variables for the
    California Housing dataset (regression).
    :return:
    """
    return fetch_california_housing(return_X_y=True)


def diabetes_regression_results():
    """
    Train and fit linear regression model on Diabetes dataset.

    Returns:
    --------
    y_test : array of shape (89,)
        The target values for the Diabetes dataset.
    y_pred : array of shape (89,)
        The predictions made by the regression model.
    """

    X, y = get_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    linear_regr = LinearRegression()
    linear_regr.fit(X_train, y_train)

    y_pred = linear_regr.predict(X_test)
    return y_test, y_pred


def california_housing_regression_results():
    """
    Train and fit linear regression model on California Housing dataset.

    Returns:
    --------
    y_test : array of shape (4128,)
        The target values for the California Housing dataset.
    y_pred : array of shape (4128,)
        The predictions made by the regression model.
    """

    X, y = get_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    linear_regr = LinearRegression()
    linear_regr.fit(X_train, y_train)

    y_pred = linear_regr.predict(X_test)
    return y_test, y_pred