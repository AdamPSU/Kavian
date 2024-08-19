import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kavian.tables.config import SEPARATOR, TABLE_LENGTH
from kavian.tables.model_stats import BinaryClassifierStatistics, RegressorStatistics
from kavian.tables.utils import format_scientific_notation, format_stat


def _add_empty_columns(table):
    empty_column = ""
    
    table.add_column(empty_column, width=25)
    table.add_column(empty_column, width=23, justify="right")
    table.add_column(empty_column, width=25)
    table.add_column(empty_column, width=23, justify="right")


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

    empty = ("", "")
    default = [empty] * available_space # Initialize a list with no entries

    if len(entries) > available_space:
        raise ValueError(
            f"Too many entries provided. Expected at most {available_space} entries, "
            f"but got {len(entries)} instead."
        )

    for key, _ in entries:
        pass

    for idx in range(len(entries)):
        default[idx] = entries[idx]

    return default


class BaseSummary:
    """Base class for all Kavian summarizers."""
    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

        self.model_name = type(self.estimator).__name__
        self.date = pd.Timestamp.now().normalize().strftime('%B %d, %Y')

        self.console = Console()


    def make_entries(self):
        """
        Create new (key, value) entries. This method is designed to be overridden
        by subclasses to provide specific implementation.
        """

        return []


class BaseRegressorSummary(BaseSummary, ABC):
    """Base class for regression summaries. All regression summaries inherit this."""

    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        self.stats = RegressorStatistics(self.estimator, self.X, self.y)


    @abstractmethod
    def summary(self):
        """Summarize Model."""


    def print_model_diagnostic(self):
        """
        Print test diagnostics below the regression summary. Currently, this method supports
        basic asssumption tests pertinent to residual analysis, and is designed to be overriden
        by subclasses to provide specific implementation.
        """

        stats = self.stats

        skew = format_stat(stats.skew())
        cond_no = format_scientific_notation(stats.cond_no())
        durbin_watson = format_stat(stats.durbin_watson())
        breusch_pagan_pval = format_stat(stats.breusch_pagan_pvalue())

        print(f"Skew: {skew} • Breusch-Pagan p-val: {breusch_pagan_pval}"
              f" • Durbin-Watson: {durbin_watson} • Cond. No. {cond_no}".center(TABLE_LENGTH))


    def create_table(self, *model_entries):
        """
        Generates a basic regression table. This method is designed to be overridden
        by subclasses to provide specific implementations.

        The regression table includes various statistics and metrics related to the
        regression analysis. The specific entries and their contents should
        be detailed in the subclass implementation.

        Parameters:
        - (Specify any new entries used by the subclass implementation, as long as they don't
           exceed the available space provided in the table)

        :return: Table
            A rich Table object containing the regression table with relevant statistics.
        """

        stats = self.stats

        model_table = Table(show_header=True, box=None, style="bold", expand=True)
        _add_empty_columns(model_table)

        # Format statistics
        log_likelihood = format_stat(stats.log_likelihood())
        aic = format_stat(stats.aic())
        bic = format_stat(stats.bic())
        r2 = format_stat(stats.r2())
        adj_r2 = format_stat(stats.adj_r2())
        mae = format_stat(stats.mae())
        rmse = format_stat(stats.rmse())

        # Other
        num_obs = str(stats.n)
        num_features = str(stats.p)
        
        # add_row() accepts 4 renderables, note that the 3rd and 4th
        # are reserved for the second column and are otherwise left empty
        model_table.add_row("Model: ", self.model_name,
                            "Log-Likelihood: ", log_likelihood)
        model_table.add_row("Date: ", self.date,
                            "AIC: ", aic)
        model_table.add_row("Dep. Variable: ", self.y_name(),
                            "BIC: ", bic)
        model_table.add_row("No. Observations: ", num_obs,
                            SEPARATOR)

        entries = include_new_entries(model_entries, available_space=5)

        (custom_entry_1, custom_value_1), (custom_entry_2, custom_value_2), \
        (custom_entry_3, custom_value_3), (custom_entry_4, custom_value_4), \
        (custom_entry_5, custom_value_5) = entries

        model_table.add_row("No. Features: ", num_features,
                            custom_entry_1, custom_value_1)
        model_table.add_row("R²: ", r2,
                            custom_entry_2, custom_value_2)
        model_table.add_row("Adj. R²: ", adj_r2,
                            custom_entry_3, custom_value_3)
        model_table.add_row("MAE: ", mae,
                            custom_entry_4, custom_value_4)
        model_table.add_row("RMSE: ", rmse,
                            custom_entry_5, custom_value_5)

        # Add an empty row
        model_table.add_row()

        return model_table


    def y_name(self):
        """Returns the name of the response variable y."""

        y_name = self.y.name if hasattr(self.y, 'name') else 'NaN'

        return y_name


def _is_binary_classifier(num_classes):
    if num_classes == 2:
        return True

    return False


class BaseBinaryClassifierSummary(BaseSummary, ABC):
    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        self.num_classes = len(np.unique(self.y))
        self.stats = BinaryClassifierStatistics(estimator, X, y)


    @abstractmethod
    def summary(self):
        """Summarize the model."""


    def create_table(self, *model_entries):
        """
        Generates a basic regression table. This method is designed to be overridden
        by subclasses to provide specific implementations.

        The regression table includes various statistics and metrics related to the
        regression analysis. The specific entries and their contents should
        be detailed in the subclass implementation.

        Parameters:
        - (Specify any new entries used by the subclass implementation, as long as they don't
           exceed the available space provided in the table)

        :return: Table
            A rich Table object containing the regression table with relevant statistics.
        """

        stats = self.stats

        model_table = Table(show_header=True, box=None, style="bold", expand=True)
        _add_empty_columns(model_table)

        # Format statistics
        accuracy = format_stat(stats.accuracy())
        recall = format_stat(stats.recall)
        precision = format_stat(stats.precision)
        f1_score = format_stat(stats.f1_score())
        roc_auc = format_stat(stats.roc_auc_score())

        # Other
        num_classes = str(self.num_classes)
        num_obs = str(self.y.shape[0])
        num_features = str(self.X.shape[1])

        # add_row() accepts 4 renderables, note that the 3rd and 4th
        # are reserved for the second column and are otherwise left empty
        model_table.add_row("Model: ", self.model_name,
                            "ROC-AUC: ", roc_auc)
        model_table.add_row("Date: ", self.date,
                            SEPARATOR)

        entries = include_new_entries(model_entries, available_space=7)

        (custom_entry_1, custom_value_1), (custom_entry_2, custom_value_2), \
        (custom_entry_3, custom_value_3), (custom_entry_4, custom_value_4), \
        (custom_entry_5, custom_value_5), (custom_entry_6, custom_value_6), \
        (custom_entry_7, custom_value_7) = entries

        model_table.add_row("No. Labels: ", num_classes,
                            custom_entry_1, custom_value_1)
        model_table.add_row("No. Obs. ", num_obs,
                            custom_entry_2, custom_value_2)
        model_table.add_row("No. Features: ", num_features,
                            custom_entry_3, custom_value_3)
        model_table.add_row("Accuracy: ", accuracy,
                            custom_entry_4, custom_value_4)
        model_table.add_row("Recall: ", recall,
                            custom_entry_5, custom_value_5)
        model_table.add_row("Precision: ", precision,
                            custom_entry_6, custom_value_6)
        model_table.add_row("F1: ", f1_score,
                            custom_entry_7, custom_value_7)

        # Add an empty row
        model_table.add_row()

        return model_table


class SimpleRegressorSummary(BaseRegressorSummary):
    """
    Simple summary table displaying useful statistics
    for Linear Regression models.
    """

    def summary(self):
        model_table = self.create_table()

        self.console.print(Panel(model_table, title="Regression Results",
                                 subtitle="Test Diagnostics"))
        self.print_model_diagnostic()


class SimpleClassifierSummary(BaseBinaryClassifierSummary):
    def summary(self):
        model_table = self.create_table()

        self.console.print(Panel(model_table, title="Classification Results",
                                 subtitle="Test Diagnostics"))





























