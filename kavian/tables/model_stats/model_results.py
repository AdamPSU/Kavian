"""
This module provides functions and utilities for calculating and summarizing
statistics useful in regression analysis. It includes methods for evaluating
various aspects of regression models, such as:

- Model diagnostics (e.g., residuals, autocorrelation, skewness)
- Performance metrics (e.g., R-squared, Adjusted R-squared, AIC, BIC)
- Error measures (e.g., Mean Squared Error, Root Mean Squared Error)
- Statistical tests (e.g., Durbin-Watson test, Omnibus test)
"""

import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix, roc_auc_score


class RegressorStatistics:
    def __init__(self, estimator, X, y):
        self.X, self.y = np.array(X), np.array(y)
        self.intercept = estimator.intercept_

        self.y_pred = estimator.predict(X)
        self.resid = self.y - self.y_pred
        self.squared_resid = self.resid**2
        self.rss = np.sum(self.squared_resid)
        self.tss = np.sum((self.y - self.y.mean())**2)

        self.n, self.p = X.shape[0], X.shape[1]


    def rss(self):
        """Returns Residual Sum of Squares"""

        return self.rss


    def tss(self):
        """Returns Total Sum of Squares"""

        return self.tss


    def r2(self):
        """Returns R-squared"""

        r2 = 1 - self.rss / self.tss

        return r2


    def adj_r2(self):
        """Returns Adjusted R-squared"""

        constant = 0
        has_intercept = self.intercept

        if has_intercept:
            constant = 1

        adj_r2 = 1 - (self.n + constant - 1) / (self.n - self.p - 1) * (1 - self.r2())

        return adj_r2


    def rmse(self):
        """Returns Root Mean Squared Error"""

        mse = np.mean(self.squared_resid)
        rmse = np.sqrt(mse)
        return rmse


    def mae(self):
        """Returns Mean Absolute Error"""

        abs_resid = np.abs(self.resid)
        mae = np.mean(abs_resid)

        return mae


    def log_likelihood(self):
        """Returns log likelihood."""

        log_likelihood = (-self.n / 2) * np.log(2 * np.pi * self.rss / self.n) - (self.n / 2)

        return log_likelihood


    def aic(self):
        """Returns the Akaike Information Criterion."""

        num_of_params = self.p
        has_intercept = self.intercept

        if has_intercept:
            num_of_params += 1

        log_likelihood = self.log_likelihood()

        aic = (2 * num_of_params) - (2 * log_likelihood)
        return aic


    def bic(self):
        """Returns the Bayesian Information Criterion."""

        num_of_params = self.p
        has_intercept = self.intercept

        if has_intercept:
            num_of_params += 1

        log_likelihood = self.log_likelihood()

        bic = (np.log(self.n) * num_of_params) - (2 * log_likelihood)
        return bic


    def skew(self):
        """
        Calculate and return the skewness of the residuals from the regression model.

        Skewness measures the asymmetry of the distribution of residuals. It indicates
        whether the residuals are skewed to the left or right of the mean.
        - A skewness of 0 indicates a symmetric distribution.
        - A positive skewness indicates a distribution with a long right tail.
        - A negative skewness indicates a distribution with a long left tail.

        Returns:
            np.float: The skewness of the residuals.
        """

        bias_corrector = self.n/((self.n - 1)*(self.n - 2))

        resid_mean = self.resid.mean()
        resid_stdev = self.resid.std(ddof=1)
        standardized_resid = (self.resid - resid_mean) / resid_stdev

        skewness = bias_corrector * np.sum(standardized_resid**3)

        return skewness


    def cond_no(self):
        """
        Calculate and return the Condition Number of the data matrix X.

        The Condition Number is a measure of the sensitivity of a model to new
        data. It is used to diagnose multicollinearity in regression and classification
        analysis. A high Condition Number indicates that the matrix is close to being
        singular, which can lead to numerical instability and unreliable results in computations.

        Returns:
            np.float: The Condition Number of the data matrix X.
        """

        cond_no = np.linalg.cond(self.X)

        return cond_no


    def breusch_pagan_pvalue(self):
        """
        Calculate and return the p-value of the Breusch-Pagan test for heteroscedasticity.

        The Breusch-Pagan test is used to detect the presence of heteroscedasticity
        in a regression model, where the variance of the residuals is not constant.
        A low p-value (for us, 0.05) indicates the presence of heteroscedasticity.

        Returns:
            np.float: The p-value of the Breusch-Pagan test.
        """

        # TODO: reimplement this method from scratch to avoid having to fit another regression model

        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan

        X_with_constant = sm.add_constant(self.X)
        _, bp_pval, _, _ = het_breuschpagan(resid=self.resid, exog_het=X_with_constant)

        return bp_pval


    def durbin_watson(self):
        """
        Calculate and return the Durbin-Watson statistic for detecting
        autocorrelation in the residuals of a linear regression model.

        The Durbin-Watson statistic ranges from 0 to 4, where:
        - A value near 2 indicates no autocorrelation.
        - A value toward 0 suggests positive autocorrelation.
        - A value toward 4 suggests negative autocorrelation.

        Returns:
        np.float: The Durbin-Watson statistic.
        """

        resid_diff = np.diff(self.resid, n=1, axis=0)
        durbin_watson = (np.sum(resid_diff**2))/self.rss

        return durbin_watson


def _divide(numerator, denominator):
    """
    Performs division & handles Zero Division errors by replacing
    errors with zero, a common convention with classifier metrics.
    """

    if denominator == 0:
        return 0 # Instead of ZeroDivision Error

    division = np.divide(numerator, denominator)

    return division


class ClassifierStatsBase(ABC):
    def __init__(self, estimator, X, y):
        self.X, self.y = np.array(X), np.array(y)
        self.intercept = estimator.intercept_

        self.y_pred = estimator.predict(self.X)
        self.y_proba = estimator.predict_proba(self.X)

        self.CM = confusion_matrix(self.y, self.y_pred)


    @abstractmethod
    def accuracy(self):
        """Accuracy of the model."""


    @abstractmethod
    def recall(self):
        """Precision of the model."""


    @abstractmethod
    def precision(self):
        """Precision of the model."""


    @abstractmethod
    def f1_score(self):
        """Harmonic mean of the model."""


    def roc_auc_score(self):
        """"""

        roc_auc = roc_auc_score(self.y, self.y_proba[:, 1])

        return roc_auc


class BinaryClassifierStatistics(ClassifierStatsBase):
    def __init__(self, estimator, X, y):
        super().__init__(estimator, X, y)

        # Flatten to a 1D-array and unpack
        self.true_neg, self.false_pos, \
        self.false_neg, self.true_pos = self.CM.ravel()

        self.recall = self.recall()
        self.precision = self.precision()


    def accuracy(self):
        """
        Calculates the accuracy of the classifier model. Accuracy finds the proportion
        of correctly specified labels (true positives & true negatives) out of all
        predictions for a response vector y.
        """

        num_of_correct = self.true_pos + self.true_neg
        accuracy = num_of_correct / len(self.y_pred)

        return accuracy


    def recall(self):
        """
        Calculates the recall of the classifier model, or the ability for the model
        to correctly identify all positive labels in the dataset.

        Recall finds the proportion of correctly specified positives out of the
        entire positive sample space of a response vector y.
        """

        total_pos = self.true_pos + self.false_neg

        # Handle zero division
        recall = _divide(self.true_pos, total_pos)

        return recall


    def precision(self):
        """
        Caculates the precision of the classifier model, or the proportion of
        predicted positives that were actually positive.

        To do so, simply express the number of true positives as a proportion
        of the sample space of our model's positive predictions.
        """

        total_pos_predictions = self.true_pos + self.false_pos

        # Handle zero division
        precision = _divide(self.true_pos, total_pos_predictions)

        return precision


    def f1_score(self):
        """
        Calculates harmonic mean of precision & recall.

        Because precision and recall are bounded by a trade-off, the
        F1 Score strikes a balance between the two measures. This metric
        is particularly useful in the presence of an unbalanced dataset.
        """

        # Handle zero division
        f1_score = _divide(2 * self.precision * self.recall, self.precision + self.recall)

        return f1_score


    def mcc(self):
        """
        Calculates Matthew's Correlation Coefficient.

        MCC is particularly helpful for binary classification. It takes
        all outputs from the confusion matrix into account, resulting in
        a balanced measure that can be used even when the classes are of
        starkly different sizes. Matthew's Correlation Coefficient mirrors
        that of the Pearson Correlation Coefficient, such that:

        - A value near 1 corresponds to near-perfect predictions
        - A value near 0 corresponds to a model no better than
          random guessing
        - A value near -1 corresponds to a model whose predictions
          are always incorrect.
        """

        tp, fp = self.true_pos, self.false_pos
        tn, fn = self.true_neg, self.false_neg

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # Handle zero division
        mcc = _divide(numerator, denominator)

        return mcc



















