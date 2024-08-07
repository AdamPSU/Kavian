from kavian.tables.base import SimpleRegressorSummary, SimpleClassifierSummary
from kavian.tables.linear_model import RegularizedRegressionSummary
from sklearn.base import ClassifierMixin, RegressorMixin

MODEL_MAPPING = {
    "Lasso": "Regularization",
    "Ridge": "Regularization",
    "ElasticNet": "Regularization",
    "LassoLars": "Regularization",
    "LassoLarsIC": "Regularization",
    "LogisticRegression": "Classification"
}

def _get_summary(estimator, X, y):
    """Factory function to return appropriate tables object."""

    estimator_name = type(estimator).__name__
    model_type = MODEL_MAPPING.get(estimator_name)

    if model_type == 'Regularization':
        return RegularizedRegressionSummary(estimator, X, y)
    elif model_type == 'Classification':
        return SimpleClassifierSummary(estimator, X, y)
    else:
        return SimpleRegressorSummary(estimator, X, y)


def summary(estimator, X, y):
    """
    Summarize a fitted Scikit-Learn model.

    Supported models include:

    - Linear Models
    - Binary Classification Models

    And more on the way.
    """

    summ = _get_summary(estimator, X, y)

    return summ.summary()