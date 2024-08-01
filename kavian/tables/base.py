import pandas as pd

from abc import ABC, abstractmethod

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kavian.tables.model_stats import RegressorStatistics
from kavian.tables.utils import format_stat, format_scientific_notation
from kavian.tables.config import TABLE_LENGTH, SEPARATOR

def _add_empty_columns(table):
    empty_column = ""
    table.add_column(empty_column)
    table.add_column(empty_column, justify="right")
    table.add_column(empty_column)
    table.add_column(empty_column, justify="right")


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

    for idx in range(len(entries)):
        default[idx] = entries[idx]

    return default


class BaseRegressorSummary(ABC):
    """Base class for regression summaries. All regression summaries inherit this."""

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
        """Create new (key, value) entries. This method is designed to be overridden
        by subclasses to provide specific implementation."""

        return []


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

        # Breusch-Pagan test
        bp_pval = format_stat(stats.breusch_pagan_pvalue())

        print(f"Skew: {skew} • Breusch-Pagan p-val: {bp_pval} • Durbin-Watson: {durbin_watson}"
              f" • Cond. No. {cond_no}".center(TABLE_LENGTH))


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
        """Returns the model's name."""

        model_name = type(self.estimator).__name__

        return model_name


    def current_date(self):
        """Returns the date."""

        date = pd.Timestamp.now().normalize().strftime('%B %d, %Y')

        return date


class BaseClassifierSummary(ABC):
    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

        self.stats = RegressorStatistics(self.estimator, self.X, self.y)
        self.console = Console()


class SimpleRegressorSummary(BaseRegressorSummary):
    """
    Simple summary table displaying useful statistics
    for Linear Regression models.
    """

    def summary(self):
        model_entries = self.make_entries()
        model_table = self.create_table(*model_entries)

        self.console.print(Panel(model_table, title="Regression Results",
                                 subtitle="Test Diagnostics"))
        self.print_model_diagnostic()




























