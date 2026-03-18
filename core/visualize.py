import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.forecast import fit_linear_degradation

warnings.filterwarnings("ignore")


def _iso_zone_label(status_text):
    if "Zone A" in status_text:
        return "A"
    if "Zone B" in status_text:
        return "B"
    if "Zone C" in status_text:
        return "C"
    if "Zone D" in status_text:
        return "D"
    return "-"


def _iso_zone_points(status_text):
    zone = _iso_zone_label(status_text)
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(zone, 0)


def _plot_velocity_timeline(logs_table, out_dir, warn_limit, crit_limit):
    equipments = sorted(logs_table["Equipment"].unique())
    fig, axes = plt.subplots(
        len(equipments),
        1,
        figsize=(12, max(4.0, 3.5 * len(equipments))),
        sharex=True,
    )
    if len(equipments) == 1:
        axes = [axes]

    for idx, (ax, equipment) in enumerate(zip(axes, equipments)):
        ordered = logs_table.loc[logs_table["Equipment"] == equipment].sort_values("Timestamp")
        ax.plot(
            ordered["Timestamp"],
            ordered["Velocity_RMS"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color="#1f77b4",
            label="Measured RMS",
        )

        fit = fit_linear_degradation(ordered)
        trend_state = "Insufficient data"
        if fit is not None:
            trend_line = fit["slope_per_day"] * fit["time_days"] + fit["intercept"]
            ax.plot(
                ordered["Timestamp"],
                trend_line,
                linestyle="--",
                linewidth=1.8,
                color="#1f4e79",
                alpha=0.95,
                label="Trend fit",
            )

            if fit["slope_per_day"] > 0:
                trend_state = "Degrading"
                days_to_crit = (crit_limit - fit["intercept"]) / fit["slope_per_day"]
                last_day = float(fit["time_days"][-1])
                if days_to_crit > last_day:
                    # Project from latest point to either threshold crossing or 8 months horizon.
                    horizon_day = min(float(days_to_crit), last_day + 240.0)
                    future_days = pd.Series([last_day, horizon_day])
                    future_dates = fit["start_date"] + pd.to_timedelta(future_days, unit="D")
                    future_values = fit["slope_per_day"] * future_days + fit["intercept"]
                    ax.plot(
                        future_dates,
                        future_values,
                        linestyle=":",
                        linewidth=2.2,
                        color="#c0392b",
                        label="Forecast",
                    )
                    failure_date = fit["start_date"] + pd.Timedelta(days=float(days_to_crit))
                    ax.axvline(failure_date, color="#c0392b", linestyle=":", linewidth=1.2, alpha=0.7)
                    ax.text(
                        failure_date,
                        crit_limit * 1.02,
                        failure_date.strftime("Est %Y-%m-%d"),
                        color="#c0392b",
                        fontsize=8,
                        rotation=90,
                        va="bottom",
                        ha="right",
                    )
            else:
                trend_state = "Stable or Improving"

        latest = ordered.iloc[-1]
        ax.scatter(latest["Timestamp"], latest["Velocity_RMS"], color="#111111", s=28, zorder=5)
        ax.annotate(
            f"{latest['Velocity_RMS']:.2f} mm/s",
            xy=(latest["Timestamp"], latest["Velocity_RMS"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8.5,
            color="#111111",
        )

        ax.axhline(warn_limit, color="#f39c12", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axhline(crit_limit, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_ylabel("RMS (mm/s)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{equipment} | Trend: {trend_state}", fontsize=10.5, loc="left")
        if idx == len(equipments) - 1:
            ax.set_xlabel("Date")

    handles = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2.0, marker="o", markersize=4, label="Measured RMS"),
        plt.Line2D([0], [0], color="#1f4e79", linewidth=1.8, linestyle="--", label="Trend fit"),
        plt.Line2D([0], [0], color="#c0392b", linewidth=2.2, linestyle=":", label="Forecast"),
        plt.Line2D([0], [0], color="#f39c12", linewidth=1.2, linestyle="--", label="ISO warning"),
        plt.Line2D([0], [0], color="#c0392b", linewidth=1.2, linestyle="--", label="ISO critical"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Velocity RMS Trend by Machine", fontsize=14, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(os.path.join(out_dir, "plot_1_vibration_timeline.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_latest_dashboard(logs_table, out_dir, crit_limit):
    latest_idx = logs_table.groupby("Equipment")["Timestamp"].idxmax()
    latest_records = logs_table.loc[latest_idx].copy().sort_values("Velocity_RMS", ascending=False)

    anomaly_ref = max(float(latest_records["Anomaly_Score"].max()), 1e-3)
    reference_map = {
        "Velocity RMS / Crit": ("Velocity_RMS", crit_limit),
        "Peak Accel / 2.0g": ("Peak_Acceleration_G", 2.0),
        "Kurtosis / 3.0": ("Kurtosis", 3.0),
        "Crest / 3.0": ("Crest_Factor", 3.0),
        "Dom Freq Amp / 0.5": ("Dominant_Frequency_Amplitude", 0.5),
        "Anomaly / Max": ("Anomaly_Score", anomaly_ref),
    }

    ratio_table = pd.DataFrame(index=latest_records["Equipment"], columns=reference_map.keys(), dtype=float)
    annot_table = pd.DataFrame(index=latest_records["Equipment"], columns=reference_map.keys(), dtype=object)

    for _, row in latest_records.iterrows():
        equipment = row["Equipment"]
        for col_label, (source_col, ref_value) in reference_map.items():
            raw_value = float(row[source_col])
            ratio_value = raw_value / ref_value if ref_value > 0 else 0.0
            ratio_table.loc[equipment, col_label] = ratio_value
            if source_col == "Anomaly_Score" or source_col == "Dominant_Frequency_Amplitude":
                annot_table.loc[equipment, col_label] = f"{raw_value:.3f}"
            else:
                annot_table.loc[equipment, col_label] = f"{raw_value:.2f}"

    max_ratio = max(1.2, float(ratio_table.max().max()) * 1.05)

    plt.figure(figsize=(12, 6.0))
    ax = sns.heatmap(
        ratio_table,
        annot=annot_table,
        fmt="",
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=max_ratio,
        center=1.0,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Ratio to reference ( > 1.0 = concern )"},
    )
    ax.set_title("Latest Feature Dashboard (text = raw value)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Health Indicators")
    ax.set_ylabel("Equipment")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_2_feature_anomaly_map.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _plot_monthly_risk(logs_table, out_dir, crit_limit):
    frame = logs_table.copy()
    frame["Month_Year"] = frame["Timestamp"].dt.strftime("%Y-%m")
    frame["Velocity_Ratio"] = frame["Velocity_RMS"] / crit_limit
    frame["Zone_Label"] = frame["ISO_Status"].apply(_iso_zone_label)
    frame["Alert_Points"] = (
        frame["ISO_Status"].apply(_iso_zone_points)
        + frame["Stat_Alert"].astype(int)
        + frame["Isolation_Anomaly"].astype(int)
    )

    equipment_order = sorted(frame["Equipment"].unique())
    month_order = sorted(frame["Month_Year"].unique())

    velocity_ratio = (
        frame.pivot_table(index="Equipment", columns="Month_Year", values="Velocity_Ratio", aggfunc="mean")
        .reindex(index=equipment_order, columns=month_order)
    )
    velocity_raw = (
        frame.pivot_table(index="Equipment", columns="Month_Year", values="Velocity_RMS", aggfunc="mean")
        .reindex(index=equipment_order, columns=month_order)
    )
    alert_points = (
        frame.pivot_table(index="Equipment", columns="Month_Year", values="Alert_Points", aggfunc="max")
        .reindex(index=equipment_order, columns=month_order)
    )
    zone_last = (
        frame.sort_values("Timestamp")
        .groupby(["Equipment", "Month_Year"])["Zone_Label"]
        .last()
        .unstack()
        .reindex(index=equipment_order, columns=month_order)
    )

    velocity_annot = velocity_raw.applymap(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    alert_annot = alert_points.copy()
    for equipment in alert_annot.index:
        for month in alert_annot.columns:
            value = alert_points.loc[equipment, month]
            zone = zone_last.loc[equipment, month]
            if pd.isna(value):
                alert_annot.loc[equipment, month] = ""
            else:
                alert_annot.loc[equipment, month] = f"{int(value)} ({zone})"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})

    sns.heatmap(
        velocity_ratio.fillna(0.0),
        mask=velocity_ratio.isna(),
        annot=velocity_annot,
        fmt="",
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=max(1.2, float(velocity_ratio.max().max()) * 1.05 if not velocity_ratio.empty else 1.2),
        center=1.0,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Velocity RMS / ISO critical"},
        ax=axes[0],
    )
    axes[0].set_title("Monthly Velocity Severity (text = RMS mm/s)", fontsize=12.5, fontweight="bold", pad=10)
    axes[0].set_ylabel("Equipment")
    axes[0].set_xlabel("")

    sns.heatmap(
        alert_points.fillna(0.0),
        mask=alert_points.isna(),
        annot=alert_annot,
        fmt="",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=5.0,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Alert points (zone + statistical + IForest)"},
        ax=axes[1],
    )
    axes[1].set_title("Monthly Alert Scoreboard (text = points and ISO zone)", fontsize=12.5, fontweight="bold", pad=10)
    axes[1].set_ylabel("Equipment")
    axes[1].set_xlabel("Month")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_3_anomaly_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_visualizations(logs_table, out_dir):
    print("\n[-] Generating feature and anomaly visualizations...")
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    for legacy_name in ["plot_2_initial_vs_latest.png", "plot_3_growth_heatmap.png"]:
        legacy_path = os.path.join(out_dir, legacy_name)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    warn_limit = float(os.environ.get("LVL_ALERT", "4.5"))
    crit_limit = float(os.environ.get("LVL_CRIT", "7.1"))

    _plot_velocity_timeline(logs_table, out_dir, warn_limit, crit_limit)
    _plot_latest_dashboard(logs_table, out_dir, crit_limit)
    _plot_monthly_risk(logs_table, out_dir, crit_limit)

    print(f"[+] 3 charts generated inside -> {out_dir}/")
