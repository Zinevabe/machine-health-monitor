import os
import numpy as np
import pandas as pd


def fit_linear_degradation(hist_data, target_col="Velocity_RMS"):
    ordered = hist_data.sort_values("Timestamp")
    if len(ordered) < 2:
        return None

    start_date = ordered["Timestamp"].iloc[0]
    time_days = np.array(
        [(ts - start_date).total_seconds() / 86400.0 for ts in ordered["Timestamp"]],
        dtype=float,
    )
    target_values = ordered[target_col].astype(float).to_numpy()

    if np.ptp(time_days) == 0:
        return None

    try:
        recency_weight = max(1.0, float(os.environ.get("FORECAST_RECENCY_WEIGHT", "5.0")))
        fit_weights = np.linspace(1.0, recency_weight, len(time_days))
        slope, intercept = np.polyfit(time_days, target_values, 1, w=fit_weights)
    except Exception:
        return None

    predictions = slope * time_days + intercept
    residual_sum = float(np.sum(fit_weights * (target_values - predictions) ** 2))
    total_sum = float(np.sum(fit_weights * (target_values - np.average(target_values, weights=fit_weights)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0

    return {
        "start_date": start_date,
        "time_days": time_days,
        "values": target_values,
        "slope_per_day": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def calc_breakdown_point(hist_data, danger_mark=None, target_col="Velocity_RMS"):
    if danger_mark is None:
        danger_mark = float(os.environ.get("LVL_CRIT", "7.1"))

    ordered = hist_data.sort_values("Timestamp")
    latest = ordered.iloc[-1]
    latest_value = float(latest[target_col])
    if latest_value >= danger_mark:
        return "Above failure threshold", latest["Timestamp"].strftime("%Y-%m-%d")

    fit = fit_linear_degradation(ordered, target_col=target_col)
    if fit is None:
        return "Insufficient data", None

    if fit["slope_per_day"] <= 0.0:
        return "Stable or Improving", None

    days_to_threshold = (danger_mark - fit["intercept"]) / fit["slope_per_day"]
    latest_elapsed_days = fit["time_days"][-1]
    remaining_days = float(days_to_threshold - latest_elapsed_days)

    if remaining_days < 0:
        return "Degrading", None

    try:
        failure_date = fit["start_date"] + pd.Timedelta(days=float(days_to_threshold))
        return "Degrading", failure_date.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return "Degrading", None
