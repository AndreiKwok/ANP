# Third party imports
import numpy as np

# Local applications imports
from sklearn.metrics import *


def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, X_train: np.ndarray) -> float:
    """
    Modified version of R-squared that has been adjusted
    for the number of predictors in the model.

    It increases when the new term improves the model more than
    would be expected by chance anddecreases when a
    predictor improves the model by less than expected.

    Parameters
    ----------
        y_true : array-like of shape (n_samples,)
            The target vector.

        y_pred : array-like of shape (n_samples,)
            Predicted target vector.

        X : array-like (n_samples, n_features)
            The data matrix on train set.

    Returns
    -------
        adj_r2 : ``float``
            Adjusted R².
    """
    adj_r2 = 1 - ((1 - r2_score(y_true, y_pred)) * (len(y_true) - 1)) / (
        len(y_true) - X_train.shape[1] - 1
    )
    return adj_r2


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Metrics to evaluate regression models.

    Parameters
    ----------
        y_true : array-like of shape (n_samples,)
            The target vector.

        y_pred : array-like of shape (n_samples,)
            Predicted target vector.

    Returns
    -------
        metrics : ``dict``
            Regression metrics.
    """
    metrics = {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "MedAE": round(median_absolute_error(y_true, y_pred), 2),
        "R²": round(r2_score(y_true, y_pred), 2),
        "MSE": round(mean_squared_error(y_true, y_pred), 2),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 2),
        "MAPE (%)": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        "BIAS": round((y_pred - y_true).mean(), 3),
    }
    return metrics
