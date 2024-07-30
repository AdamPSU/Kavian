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

        return 1 - self.rss / self.tss


    def adj_r2(self):
        """Returns Adjusted R-squared"""

        adj_r2 = 1 - (1 - self.r2()) * (self.n - 1) / (self.n - self.p - 1)

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
        """Returns skew."""

        bias_corrector = self.n/((self.n - 1)*(self.n - 2))

        resid_mean = self.resid.mean()
        resid_stdev = self.resid.std(ddof=1)
        standardized_resid = (self.resid - resid_mean) / resid_stdev

        skewness = bias_corrector * np.sum(standardized_resid**3)

        return skewness


    def cond_no(self):
        """Returns the Condition Number of data matrix X"""

        cond_no = np.linalg.cond(self.X)

        return cond_no


    def durbin_watson(self):
        """Returns Durbin-Watson test for Autocorrelation"""

        resid_diff = np.diff(self.resid, 1, 0)
        durbin_watson = (np.sum(resid_diff**2))/self.rss

        return durbin_watson














