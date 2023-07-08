from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import pandas as pd
from tqdm import trange

from joblib import Parallel, delayed


class ModelBootStrapper(BaseEstimator, ClassifierMixin):
    from .utils.utils import (
        _count_cpus,
        verify_object_inputs,
        verify_fit_inputs,
        verify_predict_inputs,
        verify_metrics_input,
    )
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
        A class to calculate confidence intervals for model's predictions based on bootstrapping.

        Parameters:
            estimator (object): an estimator with .fit and .predict methods, such as scikit-learn classifiers.
            n_boot (int): Number of bootstrap resamples used to estimate the confidence intervals. Default: 100
            n_folds (int): Number of folds for cross-validation. Default: 5
            agg_func (callable): Aggregation function used for combining predictions from different bootstrap models. Default: np.median
            ci (float): Size of the confidence interval that will be calculated. Float between 0 and 1, exclusive. Default: 0.95
            threshold (float): Decision threshold for binary classification. Default: 0.5
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
        Fits the model to the given input data.

        Parameters:
            X (pandas.DataFrame): The input data of shape (samples, features).
            y (pandas.Series): The target values of shape (samples,).
            n_samples (int, optional): The number of samples to use for training each model. If not provided,
                a default value of min(samples, 100000) will be used.

        Returns:
            None

        Notes:
            - This method fits the model to the provided input data and target values.
            - It uses the `_fit` method internally to train each model in parallel.
            - The number of models to train is determined by the `n_boot` attribute of the class.
            - The `n_jobs` parameter of the `Parallel` class is set to the number of available CPUs.
        """
        self.verify_fit_inputs(X, y, n_samples)

        if not n_samples:
            n_samples = min([X.shape[0], 100000]) if not n_samples else n_samples
        self.b_estimators = Parallel(n_jobs=self._count_cpus())(
            delayed(self._fit)(X, y, n_samples)
            for j in trange(self.n_boot, leave=True, desc="Training models")
        )

    def predict(self, X, sort_estimations=False):
        """
        Predicts using the fitted bootstrap model.

        Parameters:
            X (pandas.DataFrame): Input data of shape (samples, features).
            sort_estimations: Whether to sort returned dataframe based on point estimation. Default: False

        Returns:
            pandas.DataFrame: DataFrame containing predicted probabilities and confidence intervals.

        Raises:
            AssertionError: If the model has not been fitted before calling this method.
        """
        self.verify_predict_inputs(X, sort_estimations)

        preds = np.vstack([est.predict_proba(X)[:, 1] for est in self.b_estimators]).T
        preds = np.vstack(
            [
                self.agg_func(preds, axis=1),
                np.quantile(preds, self._ci, axis=1),
                np.quantile(preds, 1 - self._ci, axis=1),
            ]
        ).T

        preds = pd.DataFrame(
            preds,
            columns=["point_est", "lower_bound", "upper_bound"],
            index=X.index,
        )

        return preds.sort_values(by="point_est") if sort_estimations else preds
