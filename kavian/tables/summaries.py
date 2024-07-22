import pandas as pd

from colorama import Fore, Style


def _line_break(dashed: bool, colored: bool):
    style = '-' if dashed else '='
    style = Fore.BLUE + Style.BRIGHT + style if colored else style

    number_of_signs = 75
    print(style*number_of_signs)


def _style_text(string, colored: bool, bold: bool):
    string = Fore.RED + string if colored else string
    string = Style.BRIGHT + string if bold else string

    return string


class Summary:
    def __init__(self, estimator, X, y, verbose=True):
        self.estimator = estimator
        self.verbose = verbose
        self.X = X
        self.y = y

        feature_set = X.columns
        coefficients = estimator.coef_

    def model_frame(self, method):
        _line_break(dashed=False, colored=False)

        # General information present in any estimator
        gen_data_left = [('Model: ', [self.estimator.__class__.__name__]),
                         ('Method: ', [method]),
                         ('Dep. Variable: ', [self.y.name]),
                         ('No. Observations: ', [len(self.X)]),
                         ('No. Features: ', [self.estimator.n_features_in_])]

        return

    def feature_frame(self):
        pass

    def diagnostics_frame(self):
        pass
