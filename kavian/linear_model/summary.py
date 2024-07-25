import pandas as pd
import numpy as np
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
        self.n, self.p = X.shape[0], X.shape[1]

        self.console = Console()
        self.layout = Layout()

    def summary(self):
        raise NotImplementedError("Subclasses must implement this method")


class BaseRegressorSummary(BaseSummary):
    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        self.y_pred = self.estimator.predict(self.X)
        self.rss = np.sum((y - self.y_pred)**2)


    def summary(self):
        model_table = Table(show_header=True, box=None, style="bold", expand=True)

        model_table.add_column("")
        model_table.add_column("", justify="right")
        model_table.add_column("")
        model_table.add_column("", justify="right")

        info = self.process_summary_info()

        model_table.add_row("Model: ", info['model'], "Log-Likelihood: ", info['log_likely'])
        model_table.add_row("Date: ", info['date'], "AIC: ", info['aic'])
        model_table.add_row("Time: ", info['time'], "BIC: ", info['bic'])
        model_table.add_row("Dep. Variable: ", info['endog'])
        model_table.add_row("No. Observations: ", info['num_observ'])
        model_table.add_row("No. Features: ", info['num_features'])
        model_table.add_row("R²: ", info['r2'], "", "")
        model_table.add_row("Adj. R²: ", info['adj. r2'])

        self.console.print(Panel(model_table, title="Regression Results"))


    def process_summary_info(self):
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
            "bic": f"{self.get_bic():.3f}"
            }

        return basic_model_info


    def get_r2(self):
        r2 = r2_score(self.y, self.y_pred)
        adj_r2 = 1 - (1 - r2) * ((self.n - 1) / (self.n - self.p - 1))

        return r2, adj_r2


    def get_log_likelihood(self):
        log_likelihood = (-self.n / 2) * np.log(2 * np.pi * self.rss / self.n) - (self.n / 2)

        return log_likelihood

    def get_aic(self):
        k = self.p + 1
        # Calculate log-likelihood
        log_likelihood = self.get_log_likelihood()

        # Calculate AIC
        aic = (2 * k) - (2 * log_likelihood)
        return aic


    def get_bic(self):
        k = self.p + 1
        # Calculate log-likelihood
        log_likelihood = self.get_log_likelihood()

        # Calculate BIC
        bic = (np.log(self.n) * k) - (2 * log_likelihood)
        return bic





