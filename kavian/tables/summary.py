import pandas as pd

from rich.table import Table
from rich.panel import Panel
from typing import List, Tuple

sample_params = [('Model: ', '',),
                 ('Method: ', ''),
                 ('Fit Time: ', ''),
                 ('Dep. Variable(s): ', '')
                ]

class Summary:
    def __init__(self, X: pd.DataFrame, y: pd.Series, params: List[Tuple]):
        self.params = params
        self.X = X
        self.y = y

        self.feature_set = X.columns

    def create_summary_table(self):
        table = Table(show_header=True, box=None, style="bold", expand=True)

        model_info = self.params[0]

        # Empty cols 1, 3 represent the summary, and 2, 4 denote the values
        no_of_columns = 4
        for i in range(no_of_columns):
            table.add_column("")

        model, method, fit_time, dep_variables = (model_info[0][1], model_info[1][1],
                                                  model_info[2][1], model_info[3][1])

        num_features, num_observ = str(len(self.feature_set)), str(len(self.X))
        date = pd.Timestamp.now().normalize().strftime('%B %d, %Y')
        time = pd.Timestamp.now().time().strftime('%H:%M:%S')

        # NOTE: There can be at most 8 rows
        table.add_row('Model: ', model, None, None)
        table.add_row('Method: ', method, None, None)
        table.add_row('Fit time: ', fit_time, None, None)
        table.add_row('Date: ', date, None, None)
        table.add_row('Time: ', time, None, None)
        table.add_row("Dep. Variable(s): ", dep_variables, None, None)
        table.add_row("No. Observations: ", num_observ, None, None)
        table.add_row("No. Features: ", num_features, None, None)

        return Panel(table, title=f"{None} Results")

    def feature_frame(self):
        pass

    def diagnostics_frame(self):
        pass