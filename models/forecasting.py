import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None  # Handled by caller


def arima_forecast(
    series: np.ndarray,
    horizon: int = 7,
) -> Dict[str, Any]:
    """Fit a simple ARIMA model and return forecast with confidence intervals.

    Parameters
    ----------
    series: np.ndarray
        1D array of historical values ordered by time.
    horizon: int
        Number of steps to forecast ahead.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: forecast, confidence_lower, confidence_upper
    """
    if ARIMA is None:
        # Fallback to naive forecast with std-based CI
        mean_val = float(np.mean(series[-7:])) if len(series) >= 7 else float(np.mean(series))
        std_val = float(np.std(series[-30:])) if len(series) >= 30 else float(np.std(series))
        forecast = np.array([mean_val] * horizon)
        ci_mult = 1.96 * std_val
        lower = forecast - ci_mult
        upper = forecast + ci_mult
        return {
            "forecast": forecast,
            "confidence_lower": lower,
            "confidence_upper": upper,
        }

    # Ensure numeric and handle non-positive values gracefully
    y = pd.Series(np.asarray(series, dtype=float))

    # Simple automatic order selection heuristic
    # Use small orders to be robust and fast
    order_candidates = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (0, 1, 1)]
    best_aic = float("inf")
    best_order = (1, 1, 1)

    for order in order_candidates:
        try:
            model = ARIMA(y, order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = order
        except Exception:
            continue

    model = ARIMA(y, order=best_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit()
    fc = res.get_forecast(steps=horizon)
    mean_forecast = fc.predicted_mean.to_numpy()
    conf_int = fc.conf_int(alpha=0.05).to_numpy()

    lower = conf_int[:, 0]
    upper = conf_int[:, 1]

    return {
        "forecast": mean_forecast,
        "confidence_lower": lower,
        "confidence_upper": upper,
    }


# ------- Cross-validation and metrics -------

def rolling_cv_splits(n: int, window: int, horizon: int, step: int = 1) -> List[Tuple[slice, slice]]:
    splits = []
    start = window
    while start + horizon <= n:
        train_slice = slice(0, start)
        test_slice = slice(start, start + horizon)
        splits.append((train_slice, test_slice))
        start += step
    return splits


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_arima_cv(series: np.ndarray, horizon: int, window: int) -> Dict[str, float]:
    n = len(series)
    metrics = []
    for train_idx, test_idx in rolling_cv_splits(n, window, horizon):
        y_train = series[train_idx]
        y_test = series[test_idx]
        fc = arima_forecast(y_train, horizon=horizon)
        y_pred = np.asarray(fc["forecast"])[: len(y_test)]
        metrics.append({
            "rmse": rmse(y_test, y_pred),
            "mae": mae(y_test, y_pred),
            "mape": mape(y_test, y_pred),
        })
    # average
    if not metrics:
        return {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}
    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()}
    return avg



