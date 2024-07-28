import pandas as pd
import numpy as np

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from kavian.tables.base import RegressorSummaryMixin, ClassifierSummaryMixin

class RegularizedRegressionSummary(RegressorSummaryMixin):
    def summary(self):
        penalty = self.get_penalty()
        zeros = str(self.get_zero_coefficients())

        model_table = self.create_table([("Penalty: ", penalty),
                                         ("Zero Coefficients: ", zeros)])

        self.console.print(Panel(model_table, title="Regularized Regression Results"))


    def get_penalty(self):
        estimator_penalty_mapping = {
            'Ridge': 'L2',
            'Lasso': 'L1',
            'LassoLars': 'L1',
            'LassoLarsIC': 'L1',
            'ElasticNet': 'L1/L2'
        }

        estimator_name = type(self.estimator).__name__

        return estimator_penalty_mapping.get(estimator_name)


    def get_zero_coefficients(self):
        coefficients = self.estimator.coef_
        zeros = np.sum(coefficients == 0)

        return zeros

