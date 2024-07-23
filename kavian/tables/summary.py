from rich.console import Console
from rich.table import Table
from rich.panel import Panel



class Summary:
    def __init__(self, estimator, X, y, title=None):
        self.estimator = estimator
        self.X = X
        self.y = y

        #feature_set = X.columns
        #coefficients = estimator.coef_

    def create_summary_table(self):
        table = Table(show_header=True, box=None, style="bold", expand=True)

        # Empty cols 1, 3 represent the summary, and 2, 4 denote the values
        no_of_columns = 4
        for i in range(no_of_columns):
            table.add_column("")

        NaN = None
        # NOTE: There can be at most 8 rows
        table.add_row('Model: ', NaN, "No. Features", NaN)
        table.add_row('Method: ', NaN, "No. Features", NaN)
        table.add_row('Fit time: ', NaN, "No. Features", NaN)
        table.add_row("Dep. Variable(s): ", NaN, "Prob", NaN)
        table.add_row("No. Observations: ", NaN)
        table.add_row("No. Features: ", NaN)
        table.add_row(NaN , NaN, NaN, NaN)
        table.add_row(NaN, NaN, NaN, NaN)

        return Panel(table, title=f"{NaN} Results")


    def feature_frame(self):
        pass

    def diagnostics_frame(self):
        pass