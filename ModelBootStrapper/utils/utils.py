from multiprocessing import cpu_count
import pandas as pd
import numpy as np


def _count_cpus(self):
    try:
        return cpu_count()
    except:
        return 1


def verify_object_inputs(self, estimator, ci, n_boot, agg_func, n_folds):
    if not (
        hasattr(estimator, "fit")
        and hasattr(estimator, "predict")
        and hasattr(estimator, "predict_proba")
    ):
        raise AssertionError(
            "estimator has to have fit, predict, and predict_proba methods"
        )
    if isinstance(estimator, type):
        raise AssertionError("estimator is not a class. Did you forget ()?")

    if not isinstance(ci, float):
        raise TypeError("ci has to be a float")

    if not (0 < ci < 1):
        raise ValueError("ci has to be in the range (0, 1)")

    if not (isinstance(n_boot, int) and n_boot > 0):
        raise ValueError("n_boot must be an integer > 0")

    if not callable(agg_func):
        raise AssertionError("agg_func has to be a callable")

    if not (isinstance(n_folds, int) and n_boot > 0):
        raise ValueError("n_folds must be an integer > 0")

    return True


def verify_model_inputs(self, X, y=None, threshold=None):
    if not isinstance(X, pd.DataFrame):
        raise AssertionError("X has to be of type pd.DataFrame")

    if X.shape[0] == 0:
        raise ValueError("X length must have at least one sample")

    if np.any(y):
        if not isinstance(y, pd.Series):
            raise AssertionError("y has to be of type pd.Series")

        if set(y.unique()) != {0, 1}:
            raise ValueError("Allowed values in y are [0, 1]")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

    if threshold:
        if not (isinstance(threshold, float) and (threshold >= 0 and threshold <= 1)):
            raise ValueError("threshold must be a float between [0,1]")
    return True
