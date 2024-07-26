from abc import abstractmethod

import pandas as pd
import numpy as np
from typing import List, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def get_summary(estimator, X, y):
    """Factory function to return appropriate summary object."""
    pass


def summary(estimator, X, y):
    pass


class BaseSummary:
    """
    Base class for summaries; contains basic information relevant
    to any summary process.
    """

    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

        self.params = estimator.get_params()
        self.n, self.p = X.shape[0], X.shape[1]

        self.layout = Layout()
        self.console = Console()

    @abstractmethod
    def summary(self):
        """Summarize estimator."""


class BaseRegressorSummary(BaseSummary):
    """
    Base Class for regression summaries; contains basic
    information relevant to all regression models.
    """

    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        self.y_pred = self.estimator.predict(self.X)
        self.rss = np.sum((y - self.y_pred)**2)


    def _process_basic_summary(self):
        """
        Retrieves information necessary to make a basic summary.

        Default statistics provided include:

        - model : name of the Estimator
        - date : time the summary was obtained in {month day, year} format
        - time : precise time, in {hour : minutes : seconds} format
        - endog : response, or target value
        - num_observ : number of rows present in data matrix X
        - num_features : number of columns present in data matrix X
        - r2 : total variance explained by the model, goodness-of-fit estimate
        - adj. r2 : total variance explained by the model,
                    taking number of predictors into account
        - log_likely : log likelihood, also a goodness-of-fit estimate
        - aic : akaike information criterion
        - bic : bayesian information criterion
        - mae : mean absolute error for data matrix X
        - rmse : root mean squared error for data matrix X

        :return: dict for basic summary statistics
        """
        r2, adj_r2 = self.get_r2()

        basic_model_info = {
            "model": type(self.estimator).__name__,
            "date": pd.Timestamp.now().normalize().strftime('%B %d, %Y'),
            "time": pd.Timestamp.now().time().strftime('%H:%M:%S'),
            "endog": self.y.name if hasattr(self.y, 'name') else "Not Supported",
            "num_observ": str(self.n),
            "num_features": str(self.p),
            "r2": f"{r2:.3f}",
            "adj. r2": f"{adj_r2:.3f}",
            "log_likely": f"{self.get_log_likelihood():.3f}",
            "aic": f"{self.get_aic():.3f}",
            "bic": f"{self.get_bic():.3f}",
            "mae": f"{mean_absolute_error(self.y, self.y_pred):.3f}",
            "rmse": f"{root_mean_squared_error(self.y, self.y_pred):.3f}"
            }

        return basic_model_info


    def summary(self):
        """
        Prints summary table for regressor models.
        """

        model_table = Table(show_header=True, box=None, style="bold", expand=True)

        empty_column = ""
        model_table.add_column(empty_column)
        model_table.add_column(empty_column, justify="right")
        model_table.add_column(empty_column)
        model_table.add_column(empty_column, justify="right")

        info = self._process_basic_summary()

        # add_row() accepts 4 renderables, note that the 3rd and 4th
        # are reserved for the second column and are otherwise left empty
        model_table.add_row("Model: ", info['model'],
                            "Log-Likelihood: ", info['log_likely'])
        model_table.add_row("Date: ", info['date'],
                            "AIC: ", info['aic'])
        model_table.add_row("Time: ", info['time'],
                            "BIC: ", info['bic'])
        model_table.add_row("Dep. Variable: ", info['endog'],
                            "Train MAE: ", info['mae'])
        model_table.add_row("No. Observations: ", info['num_observ'],
                            "Train RMSE: ", info['rmse'])
        model_table.add_row("No. Features: ", info['num_features'])
        model_table.add_row("R²: ", info['r2'])
        model_table.add_row("Adj. R²: ", info['adj. r2'])

        self.console.print(Panel(model_table, title="Regression Results"))


    def get_r2(self):
        """Returns both R² and Adjusted R²."""

        r2 = r2_score(self.y, self.y_pred)
        adj_r2 = 1 - (1 - r2) * ((self.n - 1) / (self.n - self.p - 1))

        return r2, adj_r2


    def get_log_likelihood(self):
        """Returns log likelihood."""

        log_likelihood = (-self.n / 2) * np.log(2 * np.pi * self.rss / self.n) - (self.n / 2)

        return log_likelihood


    def get_aic(self):
        """Returns the Akaike Information Criterion."""

        predictors_plus_intercept = self.p + 1
        log_likelihood = self.get_log_likelihood()

        aic = (2 * predictors_plus_intercept) - (2 * log_likelihood)
        return aic


    def get_bic(self):
        """Returns the Bayesian Information Criterion."""

        predictors_plus_intercept = self.p + 1
        log_likelihood = self.get_log_likelihood()

        bic = (np.log(self.n) * predictors_plus_intercept) - (2 * log_likelihood)
        return bic


class BaseClassifierSummary(BaseSummary):
    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        self.y_pred = estimator.predict(X)
        self.y_proba = estimator.predict_proba(X)


    def summary(self):
        pass


    def process_summary_info(self):
        basic_model_info = {
            "model": type(self.estimator).__name__,
            "date": pd.Timestamp.now().normalize().strftime('%B %d, %Y'),
            "time": pd.Timestamp.now().time().strftime('%H:%M:%S'),
            "endog": self.y.name if hasattr(self.y, 'name') else "Not Supported",
            "num_observ": str(self.n),
            "num_features": str(self.p)
        }








