import pandas as pd
import numpy as np

from utils import basic_model_info

from typing import List, Tuple
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_summary(estimator, X, y):
    """Factory function to return appropriate summary object."""
    pass

def summary(estimator, X, y):
    pass

class BaseSummary:
    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

        self.params = estimator.get_params()

        self.console = Console()
        self.layout = Layout()

    def summary(self):
        raise NotImplementedError("Subclasses must implement this method")
class RegressorSummary(BaseSummary):
    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        self.y_pred = self.estimator.predict(self.X)
        self.n, self.p = len(self.X), len(self.X.columns)

        self.rss = np.sum((y - self.y_pred)**2)

    def summary(self):
        model_table = Table()
        no_of_columns = 4
        for i in range(no_of_columns):
            model_table.add_column("")

        model_info = basic_model_info(self.X, self.y)

        model_name = type(self.estimator).__name__
        date = model_info[0][1]
        time = model_info[1][1]

        no_of_observ = model_info[2][1]
        no_of_features = model_info[3][1]
        endog = model_info[4][1]

        r2, adj_r2 = self.get_r2()
        aic, bic = self.get_aic(), self.get_bic()




    def get_r2(self):
        r2 = r2_score(self.y, self.y_pred)
        adj_r2 = 1 - (1 - r2)((self.n - 1)/(self.n - self.p - 1))

        return r2, adj_r2


    def get_aic(self):
        k = self.p + 1
        # Calculate log-likelihood
        log_likelihood = (-self.n / 2) * np.log(2 * np.pi * self.rss / self.n) - (self.n / 2)

        # Calculate AIC
        aic = (2 * k) - (2 * log_likelihood)
        return aic


    def get_bic(self):
        k = self.p + 1
        # Calculate log-likelihood
        log_likelihood = (-self.n / 2) * np.log(2 * np.pi * self.rss / self.n) - self.n / 2

        # Calculate BIC
        bic = (np.log(self.n) * k) - (2 * log_likelihood)
        return bic





