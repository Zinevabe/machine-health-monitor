import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLUMNS = [
    "Velocity_RMS",
    "Acceleration_RMS_G",
    "Peak_to_Peak_G",
    "Kurtosis",
    "Crest_Factor",
    "Dominant_Frequency_Hz",
    "Spectral_Centroid_Hz",
    "Spectral_Entropy",
]


def assign_vibration_class(amplitude):
    limit_y = float(os.environ.get("LVL_WARN", "2.3"))
    limit_o = float(os.environ.get("LVL_ALERT", "4.5"))
    limit_r = float(os.environ.get("LVL_CRIT", "7.1"))

    if amplitude <= limit_y:
        return "Zone A (Green) - Newly commissioned"
    if amplitude <= limit_o:
        return "Zone B (Yellow) - Unrestricted operation"
    if amplitude <= limit_r:
        return "Zone C (Orange) - Restricted operation"

    return "Zone D (Red) - DAMAGE OCCURS"


def _apply_statistical_alerts(dataset):
    baseline_points = max(1, int(os.environ.get("BASELINE_POINTS", "2")))
    sigma_multiplier = float(os.environ.get("SIGMA_MULTIPLIER", "3.0"))

    dataset["Baseline_Count"] = 0
    dataset["RMS_Baseline_Mean"] = 0.0
    dataset["RMS_Baseline_Std"] = 0.0
    dataset["RMS_Alert_Threshold"] = 0.0
    dataset["RMS_Z_Score"] = 0.0
    dataset["Stat_Alert"] = False

    for _, group_idx in dataset.groupby("Equipment", sort=False).groups.items():
        ordered = dataset.loc[group_idx].sort_values("Timestamp")
        baseline_count = min(len(ordered), baseline_points)
        baseline_values = ordered.iloc[:baseline_count]["Velocity_RMS"].astype(float)

        baseline_mean = float(baseline_values.mean())
        baseline_std = float(baseline_values.std(ddof=0)) if baseline_count > 1 else 0.0
        alert_threshold = baseline_mean + sigma_multiplier * baseline_std

        dataset.loc[ordered.index, "Baseline_Count"] = baseline_count
        dataset.loc[ordered.index, "RMS_Baseline_Mean"] = baseline_mean
        dataset.loc[ordered.index, "RMS_Baseline_Std"] = baseline_std
        dataset.loc[ordered.index, "RMS_Alert_Threshold"] = alert_threshold

        evaluation_slice = ordered.iloc[baseline_count:]
        if evaluation_slice.empty:
            continue

        values = evaluation_slice["Velocity_RMS"].astype(float).to_numpy()
        if baseline_std > 0:
            z_scores = (values - baseline_mean) / baseline_std
        else:
            z_scores = np.where(values > baseline_mean, np.inf, 0.0)

        dataset.loc[evaluation_slice.index, "RMS_Z_Score"] = z_scores
        dataset.loc[evaluation_slice.index, "Stat_Alert"] = values > alert_threshold

    return dataset


def _apply_unsupervised_alerts(dataset):
    dataset["Anomaly_Score"] = 0.0
    dataset["Isolation_Anomaly"] = False

    if len(dataset) < 5:
        return dataset

    feature_frame = dataset[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    for col_name in feature_frame.columns:
        median_value = feature_frame[col_name].median()
        feature_frame[col_name] = feature_frame[col_name].fillna(median_value if pd.notna(median_value) else 0.0)

    contamination = min(max(1.0 / len(dataset), 0.05), 0.25)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    predictions = model.fit_predict(feature_frame)
    anomaly_scores = np.maximum(0.0, -model.decision_function(feature_frame))

    dataset["Anomaly_Score"] = anomaly_scores
    dataset["Isolation_Anomaly"] = predictions == -1
    return dataset


def _build_alert_reason(row):
    reasons = []
    if "Zone C" in row["ISO_Status"] or "Zone D" in row["ISO_Status"]:
        reasons.append("ISO severity")
    if row["Stat_Alert"]:
        reasons.append("3-sigma")
    if row["Isolation_Anomaly"]:
        reasons.append("isolation-forest")
    return ", ".join(reasons) if reasons else "baseline"


def _build_health_flag(row):
    if "Zone D" in row["ISO_Status"]:
        return "Critical"
    if "Zone C" in row["ISO_Status"] and (row["Stat_Alert"] or row["Isolation_Anomaly"]):
        return "High risk"
    if "Zone C" in row["ISO_Status"] or row["Stat_Alert"] or row["Isolation_Anomaly"]:
        return "Watchlist"
    return "Normal"


def attach_health_zones(dataset):
    dataset = dataset.copy()
    dataset["ISO_Status"] = [assign_vibration_class(v) for v in dataset["Velocity_RMS"]]
    dataset = _apply_statistical_alerts(dataset)
    dataset = _apply_unsupervised_alerts(dataset)
    dataset["Alert_Reason"] = dataset.apply(_build_alert_reason, axis=1)
    dataset["Health_Flag"] = dataset.apply(_build_health_flag, axis=1)
    return dataset
