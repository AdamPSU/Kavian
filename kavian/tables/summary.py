from kavian.tables.base import SimpleRegressorSummary, SimpleClassifierSummary
from kavian.tables.linear_model import RegularizedRegressorSummary, BinaryClassifierSummary

MODEL_MAPPING = {
    "Lasso": "Regularization",
    "Ridge": "Regularization",
    "ElasticNet": "Regularization",
    "LassoLars": "Regularization",
    "LassoLarsIC": "Regularization",
    "LogisticRegression": "Binary"
}

def _get_summary(estimator, X, y):
    """Factory function to return appropriate tables object."""

    estimator_name = type(estimator).__name__
    model_type = MODEL_MAPPING.get(estimator_name)

    if model_type == 'Regularization':
        return RegularizedRegressorSummary(estimator, X, y)
    elif model_type == 'Binary':
        return BinaryClassifierSummary(estimator, X, y)
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