from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def get_summary(estimator, X, y):
    """Factory function to return appropriate summary object."""
    pass


def summary(estimator, X, y):
    pass


class BaseSummary(ABC):
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
    Base class for regression summaries; contains basic
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

        - model : name of the estimator
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
            "endog": self.y.name if hasattr(self.y, 'name') else "Unknown",
            "num_observ": str(self.n),
            "num_features": str(self.p),
            "r2": f"{r2:.3f}",
            "adj. r2": f"{adj_r2:.3f}",
            "log_likely": f"{self.get_log_likelihood():.3f}",
            "aic": f"{self.get_aic():.3f}",
            "bic": f"{self.get_bic():.3f}",
            "mae": f"{self.get_mae():.3f}",
            "rmse": f"{self.get_rmse():.3f}"
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
                            "MAE: ", info['mae'])
        model_table.add_row("No. Observations: ", info['num_observ'],
                            "RMSE: ", info['rmse'])
        model_table.add_row("No. Features: ", info['num_features'])
        model_table.add_row("R²: ", info['r2'])
        model_table.add_row("Adj. R²: ", info['adj. r2'])

        self.console.print(Panel(model_table, title="Regression Results"))


    def get_rss(self):
        return self.rss


    def get_y_pred(self):
        return self.y_pred


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


    def get_mae(self):
        """Mean absolute error regression loss."""

        mae = mean_absolute_error(self.y, self.y_pred)

        return mae


    def get_rmse(self):
        """Root mean squared error regression loss."""

        rmse = root_mean_squared_error(self.y, self.y_pred)

        return rmse

class BaseClassifierSummary(BaseSummary):
    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        self.num_classes = len(np.unique(y))
        self.y_pred = estimator.predict(X)
        self.y_proba = estimator.predict_proba(X)


    def _process_basic_summary(self):
        """
        Retrieves information necessary to make a basic summary.

        Default statistics provided include:

        - model : name of the estimator
        - date : time the summary was obtained in {month day, year} format
        - time : precise time, in {hour : minutes : seconds} format
        - endog : response, or target value
        - num_observ : number of rows present in data matrix X
        - num_features : number of columns present in data matrix X
        - accuracy : proportion of correctly classified instances out of the total number of instances
        - precision : proportion of positive predictions that were actually positive
        - recall : proportion of actual positives that were correctly predicted
        - f1_score : Harmonic mean of precision and recall
        - roc_auc_score : Area under the receiver operating characteristic curve,
                          useful for binary classification problems

        :return: dict for basic summary statistics
        """

        basic_model_info = {
            "model": type(self.estimator).__name__,
            "date": pd.Timestamp.now().normalize().strftime('%B %d, %Y'),
            "time": pd.Timestamp.now().time().strftime('%H:%M:%S'),
            "num_classes": str(self.num_classes),
            "num_observ": str(self.n),
            "num_features": str(self.p),
            "accuracy": f"{self.get_accuracy():.3f}",
            "precision": f"{self.get_precision():.3f}",
            "recall": f"{self.get_recall():.3f}",
            "f1_score": f"{self.get_f1_score():.3f}",
            "roc_auc_score": f"{self.get_roc_auc_score():.3f}"
        }

        return basic_model_info


    def summary(self):
        """
        Prints summary table for classifier models.
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
                            "Precision: ", info['precision'])
        model_table.add_row("Date: ", info['date'],
                            "Recall: ", info['recall'])
        model_table.add_row("Time: ", info['time'],
                            "ROC-AUC: ", info['roc_auc_score'])
        model_table.add_row("Num. Classes: ", info['num_classes'])
        model_table.add_row("No. Observations: ", info['num_observ'],)
        model_table.add_row("No. Features: ", info['num_features'])
        model_table.add_row("Accuracy: ", info['accuracy'])
        model_table.add_row("F1 Score: ", info['f1_score'])

        self.console.print(Panel(model_table, title="Classification Results"))


    def get_accuracy(self):
        """Accuracy classification score."""

        accuracy = accuracy_score(self.y, self.y_pred)

        return accuracy


    def get_precision(self):
        """Returns the precision of the model's predictions."""

        precision = precision_score(self.y, self.y_pred)

        return precision


    def get_recall(self):
        """Returns the recall of the model's predictions."""

        recall = recall_score(self.y, self.y_pred)

        return recall


    def get_f1_score(self):
        """Compute the F1 score, also known as balanced F-score or F-measure."""

        f1 = f1_score(self.y, self.y_pred)

        return f1


    def get_roc_auc_score(self):
        """
        Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
        from prediction scores.
        """

        roc_auc = roc_auc_score(self.y, self.y_pred)

        return roc_auc





















