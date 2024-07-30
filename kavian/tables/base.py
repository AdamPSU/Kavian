from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from rich.text import Text

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

SEPARATOR = Text("-"*15, style=Style(color="red", bold=True))
TABLE_LENGTH = 79

def _add_empty_columns():
    model_table = Table(show_header=True, box=None, style="bold", expand=True)

    empty_column = ""
    model_table.add_column(empty_column)
    model_table.add_column(empty_column, justify="right")
    model_table.add_column(empty_column)
    model_table.add_column(empty_column, justify="right")

    return model_table


def include_new_entries(entries, available_space):
    """
    Takes subclass model statistics and prepares them for summary inclusion.

    Parameters:
    - entries (list of tuples): Entries to be used for analysis.
    - available_space (int): The maximum number of entries that can be accommodated.
                             This number varies by model type.

    Raises:
    - ValueError: If the number of entries exceeds available space.
    """

    # Initialize a list with no entries
    empty = ("", "")
    default = [empty] * available_space

    if len(entries) > available_space:
        raise ValueError(
            f"Too many entries provided. Expected at most {available_space} entries, "
            f"but got {len(entries)} instead."
        )

    for idx in range(len(entries)):
        default[idx] = entries[idx]

    return default


class BaseSummary:
    """
    Base class for summaries; contains basic information relevant
    to any summary process.
    """

    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

        self.n, self.p = X.shape[0], X.shape[1]

        self.layout = Layout()
        self.console = Console()


    def get_pred(self):
        """Return predictions on data matrix X"""

        y_pred = self.estimator.predict(self.X)

        return y_pred


class RegressorSummaryMixin(BaseSummary, ABC):
    """
    Base class for regression summaries; contains basic
    information relevant to all regression models.
    """

    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        self.resid = self.y - self.get_pred()
        self.squared_resid = self.resid**2
        self.rss = self.get_rss()

    @abstractmethod
    def summary(self):
        """Summarize Model."""


    @abstractmethod
    def make_entries(self):
        """Create new entries not supported by this mixin."""


    def make_model_diagnostic(self):
        print(f"Skew: {self.get_skew():.3f} • Cond. No. {self.get_cond_no():.2e} • "
              f"Durbin-Watson: {self.get_durbin_watson():.3f}".center(TABLE_LENGTH))


    def process_summary_info(self):
        """
        Retrieves and prepares information for generating basic regression tables.

        This method is intended to be overridden or extended by subclasses to provide
        specific regression summaries. The default statistics included in the summary are:

        - **model**: Name of the estimator used.
        - **date**: Date when the summary was generated, formatted as {Month Day, Year}.
        - **endog**: Response variable or target value used in the regression.
        - **num_observ**: Number of observations (rows) in the data matrix X.
        - **num_features**: Number of features (columns) in the data matrix X.
        - **r2**: R-squared, representing the proportion of variance explained by the model.
        - **adj_r2**: Adjusted R-squared, accounting for the number of predictors.
        - **log_likely**: Log-likelihood, a measure of model fit.
        - **aic**: Akaike Information Criterion, used for model comparison.
        - **bic**: Bayesian Information Criterion, used for model comparison.
        - **mae**: Mean Absolute Error (MAE) of the predictions.
        - **rmse**: Root Mean Squared Error (RMSE) of the predictions.

        :return: dict:
            A dictionary containing the regression summary statistics.
        """

        model_name = type(self.estimator).__name__
        date = pd.Timestamp.now().normalize().strftime('%B %d, %Y')

        endog = self.y.name if hasattr(self.y, 'name') else "Unknown"
        num_observ, num_features = str(self.n), str(self.p)

        r2 =f"{self.get_r2():.3f}"
        adj_r2 = f"{self.get_adj_r2():.3f}"
        llh = f"{self.get_log_likelihood():.3f}"
        aic, bic = f"{self.get_aic():.3f}", f"{self.get_bic():.3f}"
        mae = f"{self.get_mae():.3f}"
        rmse = f"{self.get_rmse():.3f}"

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


    def create_table(self, *model_entries):
        """
        Generates a basic regression table. This method is designed to be overridden
        by subclasses to provide specific implementations.

        The regression table includes various statistics and metrics related to the
        regression analysis. The specific entries and their contents should
        be detailed in the subclass implementation.

        Parameters:
        - (Specify any new entries used by the subclass implementation, if applicable)

        :return: Table
            A rich Table object containing the regression table with relevant statistics.
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
                            SEPARATOR)

        entries = include_new_entries(model_entries, available_space=5)

        (custom_entry_1, custom_value_1), (custom_entry_2, custom_value_2), \
        (custom_entry_3, custom_value_3), (custom_entry_4, custom_value_4), \
        (custom_entry_5, custom_value_5) = entries

        model_table.add_row("No. Features: ", summary_dict["Number of Features"],
                            custom_entry_1, custom_value_1)
        model_table.add_row("R²: ", summary_dict["R-squared"],
                            custom_entry_2, custom_value_2)
        model_table.add_row("Adj. R²: ", summary_dict["Adjusted R-squared"],
                            custom_entry_3, custom_value_3)
        model_table.add_row("MAE: ", summary_dict["MAE"],
                            custom_entry_4, custom_value_4)
        model_table.add_row("RMSE: ", summary_dict["RMSE"],
                            custom_entry_5, custom_value_5)

        return model_table


    def get_rss(self):
        """Retrieve Residual Sum of Squares."""

        rss = np.sum(self.squared_resid)

        return rss


    def get_r2(self):
        """Returns R²."""

        rss = self.get_rss()
        tss = np.sum((self.y - np.mean(self.y))**2)

        r2 = 1 - rss/tss

        return r2


    def get_adj_r2(self):
        """Returns Adjusted R²."""

        r2 = self.get_r2()
        adj_r2 = 1 - (1 - r2) * ((self.n - 1) / (self.n - self.p - 1))

        return adj_r2


    def get_rmse(self):
        """Returns Root Mean Squared Error."""

        mse = np.mean(self.squared_resid)
        rmse = np.sqrt(mse)

        return rmse


    def get_mae(self):
        """Returns Mean Absolute Error."""

        abs_resid = np.abs(self.resid)
        mae = np.mean(abs_resid)

        return mae


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


    def get_skew(self):
        """
        Calculate the skewness of the residuals from a fitted regression model.

        Returns:
            float: The skewness of the residuals.
        :return:
        """

        bias_corrector = self.n/((self.n - 1)*(self.n - 2))

        resid_mean = self.resid.mean()
        resid_stdev = self.resid.std(ddof=1)
        standardized_resid = (self.resid - resid_mean) / resid_stdev

        skewness = bias_corrector * np.sum(standardized_resid**3)

        return skewness


    def get_cond_no(self):
        """Returns the Condition Number of data matrix X"""

        cond_no = np.linalg.cond(self.X)

        return cond_no


    def get_durbin_watson(self):
        """Returns Durbin-Watson test for Autocorrelation"""

        resid_diff = np.diff(self.resid, 1, 0)
        durbin_watson = (np.sum(resid_diff**2))/self.rss

        return durbin_watson


    def get_f_value(self):
        pass


class ClassifierSummaryMixin(BaseSummary, ABC):
    @abstractmethod
    def summary(self):
        """Summarize Model."""


    @abstractmethod
    def make_entries(self):
        """Create new entries not supported by this mixin."""


    def process_summary_info(self):
        """
        Retrieves information necessary for generating basic classification tables.
        This method is intended to be overridden or extended by subclasses to provide
        specific implementations.

        The default statistics provided include:

        - **model**: Name of the estimator used for classification.
        - **date**: Date when the summary was generated, formatted as {Month Day, Year}.
        - **time**: Precise time when the summary was generated, formatted as {Hour:Minutes:Seconds}.
        - **endog**: Response variable or target value used in the classification.
        - **num_observ**: Number of observations (rows) in the data matrix X.
        - **num_features**: Number of features (columns) in the data matrix X.
        - **accuracy**: Proportion of correctly classified instances out of the total number of instances.
        - **precision**: Proportion of positive predictions that were actually positive.
        - **recall**: Proportion of actual positives that were correctly predicted.
        - **f1_score**: Harmonic mean of precision and recall, providing a single metric that balances both.
        - **roc_auc_score**: Area under the Receiver Operating Characteristic (ROC) curve, useful for evaluating binary classification performance.

        :return: dict:
            A dictionary containing the classification summary statistics.
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


    def create_table(self, *model_entries):
        """
        Generates a basic classification table. This method is designed to be overridden
        by subclasses to provide specific implementations.

        The classification table includes various statistics and metrics related to the
        classification analysis. The specific entries and their contents should
        be detailed in the subclass implementation.

        Parameters:
        - (Specify any new entries used by the subclass implementation, if applicable)

        :return: Table:
            A rich Table object containing the classification table with relevant statistics.
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
        model_table.add_row("No. Observations: ", summary_dict["Number of Observations"],
                            SEPARATOR)

        entries = include_new_entries(entries=model_entries, available_space=3)

        (custom_entry_1, custom_value_1), (custom_entry_2, custom_value_2), \
        (custom_entry_3, custom_value_3) = entries

        model_table.add_row("No. Features: ", summary_dict["Number of Features"],
                            custom_entry_1, custom_value_1)
        model_table.add_row("Precision: ", summary_dict["Precision"],
                            custom_entry_2, custom_value_2)
        model_table.add_row("Recall: ", summary_dict["Recall"],
                            custom_entry_3, custom_value_3)

        return model_table


    def get_num_classes(self):
        """Retrieves the number of labels present in response vector y."""

        num_classes = len(np.unique(self.y))

        return num_classes


class SimpleRegressorSummary(RegressorSummaryMixin):
    def make_entries(self):
        return []


    def summary(self):
        model_entries = self.make_entries()
        model_table = self.create_table(*model_entries)

        self.console.print(Panel(model_table, title="Simple Regression Results",
                                 subtitle="Model Diagnostics"))
        self.make_model_diagnostic()


class SimpleClassifierSummary(ClassifierSummaryMixin):
    def make_entries(self):
        return []


    def summary(self):
        model_entries = self.make_entries()
        model_table = self.create_table(*model_entries)

        self.console.print(Panel(model_table, title="Classification Results",
                                 subtitle="Model Diagnostics"))




























