from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import pandas as pd
from tqdm import trange

from joblib import Parallel, delayed


class ModelBootStrapper(BaseEstimator, ClassifierMixin):
    from .utils.utils import _count_cpus, verify_object_inputs, verify_model_inputs
    from .plotting.plotting import (
        plot_predict,
        _get_plot_config,
        _choose_samples_for_plot,
    )
    from .metrics.metrics import calculate_ppv, calculate_recall, calculate_roc_auc

    def __init__(
        self,
        estimator,
        n_boot=100,
        n_folds=5,
        agg_func=np.median,
        ci=0.95,
        threshold=0.5,
    ):
        """
        A class to calibrate binary classifiers and calculate confidence intervals for the predictions.

        estimator:    an estimator with .fit and .predict methods, such as sk-learn classifiers.
        n_boot (int): Number of bootstrap resamples used to estimate the confidence intervals. default: 100
        agg_func: aggregation function. Default: median
        ci: Size of the confidence interval that will be calculated.
            float between 0 and 1, exclusive. deafult: 0.95
        """
        self.verify_object_inputs(estimator, ci, n_boot, agg_func, n_folds)

        self.estimator = estimator
        self.agg_func = agg_func
        self.ci = ci
        self._ci = (1 - self.ci) / 2
        self.n_boot = n_boot
        self.n_folds = n_folds
        self.threshold = threshold
        self.b_estimators = []

    def _fit(self, X, y, n_samples):
        b_idx = np.random.choice(X.index, size=n_samples)
        X_resampled = X.loc[b_idx]
        y_resampled = y.loc[b_idx]

        partial_estimator = CalibratedClassifierCV(
            estimator=self.estimator, cv=self.n_folds
        )
        partial_estimator.fit(X_resampled, y_resampled)
        return partial_estimator

    def fit(self, X, y, n_samples=None):
        """
        A method to fit a calibrated estimator and n_boot bootstrapped estimators.

        X: Pandas dataframe of shape (samples, features).
        y: pd.Series of shape (samples,). Binary classifications.
        """
        self.verify_model_inputs(X, y)

        if not n_samples:
            n_samples = min([X.shape[0], 100000]) if not n_samples else n_samples
        self.b_estimators = Parallel(n_jobs=self._count_cpus())(
            delayed(self._fit)(X, y, n_samples)
            for j in trange(self.n_boot, leave=True, desc="Training models")
        )

    def predict(self, X):
        """
        A method to predict a fitted bootstrap model.

        X:  Numpy array or Pandas dataframe of shape (samples, features).
        """
        # verify the class if fitted:
        self.verify_model_inputs(X)

        if not self.b_estimators:
            raise AssertionError("Must fit estimators before using predict")

        preds = np.vstack([est.predict_proba(X)[:, 1] for est in self.b_estimators]).T
        preds = np.vstack(
            [
                self.agg_func(preds, axis=1),
                np.quantile(preds, self._ci, axis=1),
                np.quantile(preds, 1 - self._ci, axis=1),
            ]
        ).T

        return pd.DataFrame(
            preds, columns=["point_est", "lower_bound", "upper_bound"], index=X.index
        ).sort_values(by="point_est")
