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


def _add_empty_columns():
    model_table = Table(show_header=True, box=None, style="bold", expand=True)

    empty_column = ""
    model_table.add_column(empty_column)
    model_table.add_column(empty_column, justify="right")
    model_table.add_column(empty_column)
    model_table.add_column(empty_column, justify="right")

    return model_table


class BaseSummary:
    """
    Base class for summaries; contains basic information relevant
    to any tables process.
    """

    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

        self.n, self.p = X.shape[0], X.shape[1]

        self.layout = Layout()
        self.console = Console()


    def get_pred(self):
        """Return prediction on data matrix X"""

        y_pred = self.estimator.predict(self.X)

        return y_pred


class RegressorSummaryMixin(BaseSummary, ABC):
    """
    Base class for regression summaries; contains basic
    information relevant to all regression models.
    """

    @abstractmethod
    def summary(self):
        """Summarize Model."""


    def process_summary_info(self):
        """
        Retrieves information necessary to make basic regression tables. This
        method is meant to be reimplemented/extended by subclasses.

        Default statistics provided include:

        - model : name of the estimator
        - date : time the tables was obtained in {month day, year} format
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

        :return: regression summary dict
        """

        model_name = type(self.estimator).__name__
        date = pd.Timestamp.now().normalize().strftime('%B %d, %Y')

        endog = self.y.name if hasattr(self.y, 'name') else "Unknown"
        num_observ, num_features = str(self.n), str(self.p)

        r2 =f"{r2_score(self.y, self.get_pred()):.3f}"
        adj_r2 = f"{self.get_adj_r2():.3f}"
        llh = f"{self.get_log_likelihood():.3f}"
        aic, bic = f"{self.get_aic():.3f}", f"{self.get_bic():.3f}"
        mae = f"{mean_absolute_error(self.y, self.get_pred()):.3f}"
        rmse = f"{root_mean_squared_error(self.y, self.get_pred()):.3f}"

        summary_dict = {
            "Model Name": model_name,
            "Date": date,
            "Endogenous Variable": endog,
            "Number of Observations": num_observ,
            "Number of Features": num_features,
            "R-squared": r2,
            "Adjusted R-squared": adj_r2,
            "Log-Likelihood": llh,
            "AIC": aic,
            "BIC": bic,
            "MAE": mae,
            "RMSE": rmse
        }

        return summary_dict


    def create_table(self, extra=None):
        """
        Basic regression table. This method is meant to be
        reimplemented by subclasses.

        Parameters include:

        extra : list of tuples containing additional rows
        """

        model_table = _add_empty_columns()
        summary_dict = self.process_summary_info()

        # add_row() accepts 4 renderables, note that the 3rd and 4th
        # are reserved for the second column and are otherwise left empty
        model_table.add_row("Model: ", summary_dict["Model Name"],
                            "Log-Likelihood: ", summary_dict["Log-Likelihood"])
        model_table.add_row("Date: ", summary_dict["Date"],
                            "AIC: ", summary_dict["AIC"])
        model_table.add_row("Dep. Variable: ", summary_dict["Endogenous Variable"],
                            "BIC: ", summary_dict["BIC"])
        model_table.add_row("No. Observations: ", summary_dict["Number of Observations"],
                            "MAE: ", summary_dict["MAE"])
        model_table.add_row("No. Features: ", summary_dict["Number of Features"],
                            "RMSE: ", summary_dict["RMSE"])

        if not extra:
            extra = [("", ""), ("", "")]

        model_table.add_row("R²: ", summary_dict["R-squared"],
                            extra[0][0], extra[0][1])
        model_table.add_row("Adj. R²: ", summary_dict["Adjusted R-squared"],
                            extra[1][0], extra[1][1])

        return model_table


    def get_rss(self):
        """Retrieve Residual Sum of Squares."""

        rss = np.sum((self.y - self.get_pred())**2)

        return rss


    def get_adj_r2(self):
        """Returns Adjusted R²"""

        r2 = r2_score(self.y, self.get_pred())
        adj_r2 = 1 - (1 - r2) * ((self.n - 1) / (self.n - self.p - 1))

        return adj_r2


    def get_log_likelihood(self):
        """Returns log likelihood."""

        log_likelihood = (-self.n / 2) * np.log(2 * np.pi * self.get_rss() / self.n) - (self.n / 2)

        return log_likelihood


    def get_aic(self):
        """Returns the Akaike Information Criterion."""

        num_of_params = self.p
        has_intercept = self.estimator.fit_intercept

        if has_intercept:
            num_of_params += 1

        log_likelihood = self.get_log_likelihood()

        aic = (2 * num_of_params) - (2 * log_likelihood)
        return aic


    def get_bic(self):
        """Returns the Bayesian Information Criterion."""

        num_of_params = self.p
        has_intercept = self.estimator.fit_intercept

        if has_intercept:
            num_of_params += 1

        log_likelihood = self.get_log_likelihood()

        bic = (np.log(self.n) * num_of_params) - (2 * log_likelihood)
        return bic


class SimpleRegressorSummary(RegressorSummaryMixin):
    def summary(self):

        model_table = super().create_table()

        self.console.print(Panel(model_table, title="Simple Regression Results"))


class ClassifierSummaryMixin(BaseSummary, ABC):
    @abstractmethod
    def summary(self):
        """Summarize Model."""


    def process_summary_info(self):
        """
        Retrieves information necessary to make basic classification tables. This
        method is meant to be reimplemented/extended by subclasses.

        Default statistics provided include:

        - model : name of the estimator
        - date : time the tables was obtained in {month day, year} format
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
        :return:
        """
        model_name = type(self.estimator).__name__
        date = pd.Timestamp.now().normalize().strftime('%B %d, %Y')

        num_classes = str(self.get_num_classes())
        num_observ, num_features = str(self.n), str(self.p)

        accuracy = f"{accuracy_score(self.y, self.get_pred()):.3f}"
        precision = f"{precision_score(self.y, self.get_pred()):.3f}"
        recall = f"{recall_score(self.y, self.get_pred()):.3f}"
        f1 = f"{f1_score(self.y, self.get_pred()):.3f}"
        roc_auc = f"{roc_auc_score(self.y, self.get_pred()):.3f}"

        summary_dict = {
            "Model Name": model_name,
            "Date": date,
            "Number of Classes": num_classes,
            "Number of Observations": num_observ,
            "Number of Features": num_features,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC Score": roc_auc
        }

        return summary_dict


    def create_table(self, extra=None):
        """
        Basic summary table. This method is meant to be
        reimplemented by subclasses.

        Parameters include:

        extra : list of tuples containing additional rows
        """

        model_table = _add_empty_columns()
        summary_dict = self.process_summary_info()

        # add_row() accepts 4 renderables, note that the 3rd and 4th
        # are reserved for the second column and are otherwise left empty
        model_table.add_row("Model: ", summary_dict["Model Name"],
                           "F1: ", summary_dict["F1 Score"])
        model_table.add_row("Date: ", summary_dict["Date"],
                            "ROC-AUC: ", summary_dict["ROC AUC Score"])
        model_table.add_row("No. Classes", summary_dict["Number of Classes"],
                            "Accuracy: ", summary_dict["Accuracy"])

        if not extra:
            extra = [("", ""), ("", ""), ("", ""), ("", "")]

        model_table.add_row("No. Observations: ", summary_dict["Number of Observations"],
                            extra[0][0], extra[0][1])
        model_table.add_row("No. Features: ", summary_dict["Number of Features"],
                            extra[1][0], extra[0][1])
        model_table.add_row("Precision: ", summary_dict["Precision"],
                            extra[2][0], extra[0][1])
        model_table.add_row("Recall: ", summary_dict["Recall"],
                            extra[3][0], extra[0][1])

        return model_table


    def get_num_classes(self):
        """Retrieves the number of labels present in response vector y."""

        num_classes = len(np.unique(self.y))

        return num_classes


class SimpleClassifierSummary(ClassifierSummaryMixin):
    def summary(self):

        model_table = super().create_table()

        self.console.print(Panel(model_table, title="Classification Results"))




























