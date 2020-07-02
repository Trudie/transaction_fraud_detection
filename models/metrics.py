from typing import Tuple
import numpy as np
import lightgbm as lgb


def weighted_cost(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> Tuple[float, float]:
    """This is a metric that measures the performance of fraud detection model

    Parameters
    ----------
    y_true: 1d array-like of shape (n_samples,)
        Actual binary label with 0 and 1
    y_pred: 1d array-like of shape (n_samples,)
        Predicted binary label with 0 and 1
    sample_weight : 1d array-like of shape (n_samples,)
        Weight for each instance, in our case, should be `amount_eur`

    Return
    ------
    metrics: float
        The value of risk metric.
    fp_rate: float
        The percentage of False Positive
    """

    fp_idx = np.intersect1d(
        np.argwhere(y_true == 0).squeeze(), np.argwhere(y_pred == 1).squeeze(),
    )
    fn_idx = np.intersect1d(
        np.argwhere(y_true == 1).squeeze(), np.argwhere(y_pred == 0).squeeze(),
    )

    fp, fn = len(fp_idx), len(fn_idx)

    if sample_weight is not None:
        sample_weight = 0.5 * sample_weight

        fp = 0.5 * np.sum(sample_weight[fn_idx])
        fn = fn * 15 + 0.5 * np.sum(sample_weight[fn_idx])

    return fp, fn


def lgb_weighted_cost(preds: np.ndarray, dtrain: lgb.Dataset) -> Tuple[str, float, bool]:
    """
    Lightgbm evaluation metrics
    """
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    wc = sum(weighted_cost(labels, preds, sample_weight=weights))
    return 'weighted cost', wc, False
