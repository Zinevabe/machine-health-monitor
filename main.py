import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from prettytable import PrettyTable

from core.classifier import attach_health_zones
from core.forecast import calc_breakdown_point
from core.parser import collect_measurements
from core.visualize import generate_visualizations


def start_job():
    base_loc = Path(__file__).resolve().parent
    load_dotenv(base_loc / ".env")

    input_dir = os.environ.get("DATA_DIR", str(base_loc / "data"))
    out_dir = os.environ.get("OUTPUT_DIR", str(base_loc / "output"))

    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "#" * 78)
    print(">>> MACHINE VIBRATION MONITOR, ANOMALY DETECTOR & RUL ESTIMATOR <<<".center(78))
    print(f"| Source Folder: {input_dir}")
    print("#" * 78 + "\n")

    logs_table = collect_measurements(input_dir)
    if logs_table.empty:
        print("[!] Warning: Zero data extracted.")
        return

    logs_table = attach_health_zones(logs_table)
    generate_visualizations(logs_table, out_dir)

    summary_list = []
    for equipment, group_df in logs_table.groupby("Equipment", sort=False):
        ordered_hist = group_df.sort_values("Timestamp")
        latest = ordered_hist.iloc[-1]
        trend_status, failure_date = calc_breakdown_point(ordered_hist)

        summary_list.append(
            {
                "Equipment": equipment,
                "Latest_Timestamp": latest["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "Latest_Velocity_RMS": round(float(latest["Velocity_RMS"]), 3),
                "Latest_Kurtosis": round(float(latest["Kurtosis"]), 3),
                "Latest_Crest_Factor": round(float(latest["Crest_Factor"]), 3),
                "ISO_Status": latest["ISO_Status"],
                "Health_Flag": latest["Health_Flag"],
                "Stat_Alert": "Yes" if bool(latest["Stat_Alert"]) else "No",
                "Isolation_Anomaly": "Yes" if bool(latest["Isolation_Anomaly"]) else "No",
                "Alert_Reason": latest["Alert_Reason"],
                "Trend": trend_status,
                "Est_Failure": failure_date if failure_date else "N/A",
            }
        )

    final_overview = pd.DataFrame(summary_list)

    table = PrettyTable()
    table.field_names = [
        "Equipment",
        "Velocity RMS",
        "Kurtosis",
        "Health",
        "3-Sigma",
        "IForest",
        "Trend",
        "Est_Failure",
    ]
    table.align = "l"
    table.align["Velocity RMS"] = "r"
    table.align["Kurtosis"] = "r"
    table.align["Trend"] = "c"
    table.align["Est_Failure"] = "c"

    for row in summary_list:
        table.add_row(
            [
                row["Equipment"],
                f"{row['Latest_Velocity_RMS']:.3f}",
                f"{row['Latest_Kurtosis']:.3f}",
                row["Health_Flag"],
                row["Stat_Alert"],
                row["Isolation_Anomaly"],
                row["Trend"],
                row["Est_Failure"],
            ]
        )

    print("\n[ LATEST ASSET OVERVIEW ]")
    print(table)

    detail_target = Path(out_dir) / "vibration_feature_history.csv"
    summary_target = Path(out_dir) / "vibration_analysis_summary.csv"
    logs_table.sort_values(["Equipment", "Timestamp"]).to_csv(detail_target, index=False)
    final_overview.to_csv(summary_target, index=False)

    print(f"\n[+] Summary exported to -> {summary_target}")
    print(f"[+] Detailed feature history exported to -> {detail_target}")


if __name__ == "__main__":
    start_job()
