from multiprocessing import cpu_count
import pandas as pd
import numpy as np


def _count_cpus(self):
    """
    Returns:
        int: The number of available CPUs.

    Notes:
        - If the number of CPUs cannot be determined, it returns 1 as a fallback.
    """
    try:
        return cpu_count()
    except:
        return 1


def verify_object_inputs(self, estimator, ci, n_boot, agg_func, n_folds):
    """
    Verifies the input parameters of the `ModelBootStrapper` class.

    Parameters:
        estimator (object): The estimator object to be verified.
        ci (float): The size of the confidence interval.
        n_boot (int): The number of bootstrap resamples.
        agg_func (callable): The aggregation function.
        n_folds (int): The number of cross-validation folds.

    Returns:
        bool: True if all inputs are valid.

    Raises:
    AssertionError: If the `estimator` does not have the required methods (`fit`, `predict`, `predict_proba`),
                    or if it is a class instead of an instance.
                    If `agg_func` is not callable.
    TypeError: If `ci` is not a float.
               If `n_boot` is not an integer.
               If `n_folds` is not an integer.
    ValueError: If `ci` is not within the range (0, 1).
                If `n_boot` is not a positive integer.
                If `n_folds` is not a positive integer.
    """
    if not (
        hasattr(estimator, "fit")
        and hasattr(estimator, "predict")
        and hasattr(estimator, "predict_proba")
    ):
        raise AssertionError(
            "estimator has to have fit, predict, and predict_proba methods"
        )
    if isinstance(estimator, type):
        raise TypeError("estimator is not a class. Did you forget ()?")

    if not isinstance(ci, float):
        raise TypeError("ci has to be a float")
    elif not (0 < ci and ci < 1):
        raise ValueError("ci has to be in the range (0, 1)")

    if not isinstance(n_boot, int):
        raise TypeError("n_boot must be an integer")
    elif n_boot <= 0:
        raise ValueError("n_boot must be an integer > 0")

    if not callable(agg_func):
        raise AssertionError("agg_func has to be a callable")

    if not isinstance(n_folds, int):
        raise TypeError("n_folds must be an integer")
    elif n_folds <= 0:
        raise ValueError("n_folds must be > 0")

    return True


def verify_fit_inputs(self, X, y, n_samples):
    """
    Verifies the input parameters of the `fit` method in the `ModelBootStrapper` class.

    Parameters:
        X (pandas.DataFrame): The input data of shape (samples, features).
        y (pandas.Series): The target values of shape (samples,).

    Returns:
        bool: True if all inputs are valid.

    Raises:
        TypeError: If `X` is not of type `pd.DataFrame`.
                   If `y` is not of type `pd.Series`.

        ValueError: If `y` contains values other than [0, 1].

        AssertionError: If `X` length does not have at least one sample.
                        If `X` and `y` do not have the same number of samples.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X has to be of type pd.DataFrame")

    if not isinstance(y, pd.Series):
        raise TypeError("y has to be of type pd.Series")

    if n_samples:
        if not isinstance(n_samples, int):
            raise TypeError("n_samples has to be of type integer")

        if n_samples <= 0 or n_samples > X.shape[0]:
            raise ValueError("n_samples has to be > 0 and <= number of samples in X")

    if set(y.unique()) != {0, 1}:
        raise ValueError("y must include only [0, 1]")

    if X.shape[0] == 0:
        raise AssertionError("X length must have at least one sample")

    if X.shape[0] != y.shape[0]:
        raise AssertionError("X and y must have the same number of samples")

    return True


def verify_predict_inputs(self, X, sort_estimations):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X has to be of type pd.DataFrame")

    if not isinstance(sort_estimations, bool):
        raise TypeError("sort_estimations has to be of type boolean")

    if not self.b_estimators:
        raise AssertionError("Please use `fit` before trying to predict")

    return True


def verify_metrics_input(self, X, y, threshold=None):
    """
    Verifies the input parameters of the metrics methods in the `ModelBootStrapper` class.

    Parameters:
        X (pandas.DataFrame): The input data of shape (samples, features).
        y (pandas.Series, np.ndarray): The target values of shape (samples,).
        threshold: Decision threshold for classification

    Returns:
        bool: True if all inputs are valid.

    Raises:
        TypeError: If `y` is not of type `pd.Series`.
                   If `threshold` is not of type float.

        ValueError: If `y` contains values other than [0, 1].
                    If `threshold` no within [0, 1]

        AssertionError: If `X` length does not have at least one sample.
                        If `X` and `y` do not have the same number of samples.
    """
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y has to be of type pd.Series")

    if threshold:
        if not isinstance(threshold, float):
            raise TypeError("threshold has to be of type float")

        if threshold < 0 or threshold > 1:
            raise ValueError("threhols has to be between [0, 1]")

    if X.shape[0] != y.shape[0]:
        raise AssertionError("X and y must have the same number of samples")

    if set(y.unique()) != {0, 1} or set(y.unique()) != {0} or set(y.unique()) != {1}:
        raise ValueError("y_true must include only [0, 1]")

    return True
