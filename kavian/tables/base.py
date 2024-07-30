from abc import ABC, abstractmethod

import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kavian.tables.model_stats import RegressorStatistics
from kavian.tables.utils import format_stat, format_scientific_notation
from kavian.tables.config import TABLE_LENGTH, SEPARATOR

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


class BaseRegressorSummary(ABC):
    """
    Base class for regression summaries; contains basic
    information relevant to all regression models.
    """

    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

        self.stats = RegressorStatistics(self.estimator, self.X, self.y)
        self.console = Console()


    @abstractmethod
    def summary(self):
        """Summarize Model."""


    def make_entries(self):
        """Create new entries not included in this Mixin."""

        return []


    def print_model_diagnostic(self):
        skew = format_stat(self.stats.skew())
        cond_no = format_scientific_notation(self.stats.cond_no())
        durbin_watson = format_stat(self.stats.durbin_watson())

        print(f"Skew: {skew} • Cond. No. {cond_no} • Durbin-Watson: {durbin_watson}".center(TABLE_LENGTH))


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

        info = self.stats
        model_table = _add_empty_columns()

        # Format statistics
        log_likelihood = format_stat(info.log_likelihood())
        aic = format_stat(info.aic())
        bic = format_stat(info.bic())
        r2 = format_stat(info.r2())
        adj_r2 = format_stat(info.adj_r2())
        mae = format_stat(info.mae())
        rmse = format_stat(info.rmse())

        # Other
        num_obs = str(info.n)
        num_features = str(info.p)
        
        # add_row() accepts 4 renderables, note that the 3rd and 4th
        # are reserved for the second column and are otherwise left empty
        model_table.add_row("Model: ", self.model_name(),
                            "Log-Likelihood: ", log_likelihood)
        model_table.add_row("Date: ", self.current_date(),
                            "AIC: ", aic)
        model_table.add_row("Dep. Variable: ", self.y_name(),
                            "BIC: ", bic)
        model_table.add_row("No. Observations: ", num_obs,
                            SEPARATOR)

        entries = include_new_entries(model_entries, available_space=5)

        (custom_entry_1, custom_value_1), (custom_entry_2, custom_value_2), \
        (custom_entry_3, custom_value_3), (custom_entry_4, custom_value_4), \
        (custom_entry_5, custom_value_5) = entries

        # Format custom statistics
        custom_value_1 = format_stat(custom_value_1)
        custom_value_2 = format_stat(custom_value_2)
        custom_value_3 = format_stat(custom_value_3)
        custom_value_4 = format_stat(custom_value_4)
        custom_value_5 = format_stat(custom_value_5)

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


    def model_name(self):
        """Returns the estimator's name."""

        model_name = type(self.estimator).__name__

        return model_name


    def current_date(self):
        """Returns the date."""

        date = pd.Timestamp.now().normalize().strftime('%B %d, %Y')

        return date


class SimpleRegressorSummary(BaseRegressorSummary):
    def summary(self):
        model_entries = self.make_entries()
        model_table = self.create_table(*model_entries)

        self.console.print(Panel(model_table, title="Simple Regression Results",
                                 subtitle="Test Diagnostics"))
        self.print_model_diagnostic()


























