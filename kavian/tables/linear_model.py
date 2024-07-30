import pandas as pd
import numpy as np

from rich.panel import Panel
from kavian.tables.base import BaseRegressorSummary

class RegularizedRegressionSummary(BaseRegressorSummary):
    def summary(self):
        model_entries = self.make_entries()
        model_table = self.create_table(*model_entries)

        self.console.print(Panel(model_table, title="Regularized Regression Results",
                                 subtitle="Test Diagnostics"))
        self.print_model_diagnostic()


    def make_entries(self):
        penalty = ("Penalty: ", self.penalty())
        sparse_coefs = ("Sparse Features: ", str(self.sparse_coefficients()))

        return [penalty, sparse_coefs]


    def penalty(self):
        estimator_penalty_mapping = {
            'Ridge': 'L2',
            'Lasso': 'L1',
            'LassoLars': 'L1',
            'LassoLarsIC': 'L1',
            'ElasticNet': 'L1/L2'
        }

        estimator_name = self.model_name()

        return estimator_penalty_mapping.get(estimator_name)


    def sparse_coefficients(self):
        coefficients = self.estimator.coef_
        zeros = np.sum(coefficients == 0)

        return zeros

