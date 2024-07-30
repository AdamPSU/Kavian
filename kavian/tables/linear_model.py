import pandas as pd
import numpy as np

from rich.panel import Panel
from kavian.tables.base import RegressorSummaryMixin

class RegularizedRegressionSummary(RegressorSummaryMixin):
    def make_entries(self):
        penalty = ("Penalty: ", self.get_penalty())
        sparse_coefs = ("Sparse Features: ", str(self.get_sparse_coefficients()))

        return [penalty, sparse_coefs]


    def summary(self):
        model_entries = self.make_entries()
        model_table = self.create_table(*model_entries)

        self.console.print(Panel(model_table, title="Regularized Regression Results",
                                 subtitle="Statistical Diagnostics"))


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


    def get_sparse_coefficients(self):
        coefficients = self.estimator.coef_
        zeros = np.sum(coefficients == 0)

        return zeros

