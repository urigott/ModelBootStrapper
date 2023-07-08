from sklearn.metrics import precision_score, recall_score, roc_auc_score
import numpy as np


def calculate_ppv(self, X, y_true, threshold=None):
    """
    Calculates the Positive Predictive Value (PPV) using the fitted bootstrap model.

    Parameters:
        X (pandas.DataFrame): Input data of shape (samples, features).
        y_true (pandas.Series): True labels of shape (samples,).
        threshold (float, optional): Decision threshold for classification. If not provided, the default threshold
                                    of the model will be used.

    Returns:
        tuple: A tuple containing the calculated PPV and its confidence interval.

    Note:
        - The PPV represents the proportion of positive predictions that are correct.
        - The confidence interval is estimated based on the bootstrap resamples of the model.

    """
    threshold = threshold if threshold else self.threshold
    self.verify_metrics_input(X, y_true, threshold)

    ppvs = [
        precision_score(y_true, est.predict_proba(X)[:, 1] > threshold)
        for est in self.b_estimators
    ]
    ppv = self.agg_func(ppvs)
    ppv_CI = (np.quantile(ppvs, self._ci), np.quantile(ppvs, 1 - self._ci))
    return ppv, ppv_CI


def calculate_recall(self, X, y_true, threshold=None):
    threshold = threshold if threshold else self.threshold

    recalls = [
        recall_score(y_true, est.predict_proba(X)[:, 1] > threshold)
        for est in self.b_estimators
    ]
    recall = self.agg_func(recalls)
    recall_CI = (np.quantile(recalls, self._ci), np.quantile(recalls, 1 - self._ci))
    return recall, recall_CI


def calculate_roc_auc(self, X, y_true):
    roc_aucs = [
        roc_auc_score(y_true, est.predict_proba(X)[:, 1]) for est in self.b_estimators
    ]
    roc_auc = self.agg_func(roc_aucs)
    roc_auc_CI = (
        np.quantile(roc_aucs, self._ci),
        np.quantile(roc_aucs, 1 - self._ci),
    )
    return roc_auc, roc_auc_CI
