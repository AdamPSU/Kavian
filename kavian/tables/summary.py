from kavian.tables.base import SimpleClassifierSummary, SimpleRegressorSummary
from kavian.tables.linear_model import RegularizedRegressionSummary
from sklearn.base import ClassifierMixin, RegressorMixin

MODEL_MAPPING = {
    "Lasso": "Regularization",
    "Ridge": "Regularization",
    "ElasticNet": "Regularization"
}

def _get_summary(estimator, X, y):
    """Factory function to return appropriate tables object."""

    estimator_name = type(estimator).__name__
    model_type = MODEL_MAPPING.get(estimator_name)

    if model_type == 'Regularization':
        return RegularizedRegressionSummary(estimator, X, y)
    elif isinstance(estimator, ClassifierMixin):
        return SimpleClassifierSummary(estimator, X, y)
    elif isinstance(estimator, RegressorMixin):
        return SimpleRegressorSummary(estimator, X, y)
    else:
        raise ValueError(f"Estimator must be either a classifier or a regressor, got {type(estimator).__name__} instead.")


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