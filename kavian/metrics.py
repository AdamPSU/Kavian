import numpy as np
import pandas

def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

