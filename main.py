import os
import pandas as pd
from pathlib import Path

# Local core routines
from core.parser import collect_measurements
from core.classifier import attach_health_zones
from core.forecast import calc_breakdown_point
from core.visualize import generate_visualizations

def start_job():
    base_loc = Path(__file__).resolve().parent
    input_dir = os.environ.get("DATA_DIR", str(base_loc / "data"))
    out_dir = os.environ.get("OUTPUT_DIR", str(base_loc / "output"))
    
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "#" * 65)
    print(">>> MACHINE VIBRATION MONITOR & PREDICTOR <<<".center(65))
    print(f"| Source Folder: {input_dir}")
    print("#" * 65 + "\n")

    logs_table = collect_measurements(input_dir)
    if logs_table.empty:
        print("[!] Warning: Zero data extracted.")
        return

    logs_table = attach_health_zones(logs_table)
    
    # ===== PLOT ALL CHARTS =====
    generate_visualizations(logs_table, out_dir)

    summary_list = []
    
    # Process each distinct machine
    for asst, group_df in logs_table.groupby("Equipment", sort=False):
        ordered_hist = group_df.sort_values("Timestamp")
        latest = ordered_hist.iloc[-1]
        
        status_trend, drop_date = calc_breakdown_point(ordered_hist)
        
        summary_list.append({
            "Equipment": asst,
            "Latest_Velocity_RMS": f"{latest['Velocity_RMS']:.2f}",
            "Status": latest["ISO_Status"],
            "Trend": status_trend,
            "Est_Failure": drop_date if drop_date else "N/A"
        })

    final_overview = pd.DataFrame(summary_list)
    
    from prettytable import PrettyTable
    pt = PrettyTable()
    pt.field_names = ["Equipment", "Velocity RMS (mm/s)", "Status", "Trend", "Est_Failure"]
    pt.align = "l"  # Align text to the left
    pt.align["Velocity RMS (mm/s)"] = "r" # Align numbers to the right
    pt.align["Trend"] = "c"
    pt.align["Est_Failure"] = "c"
    
    for row in summary_list:
        pt.add_row([row["Equipment"], row["Latest_Velocity_RMS"], row["Status"], row["Trend"], row["Est_Failure"]])
    
    print("\n[ TODAY'S ASSET OVERVIEW ]")
    print(pt)

    csv_target = Path(out_dir) / "vibration_analysis_summary.csv"
    final_overview.to_csv(csv_target, index=False)
    print(f"\n[+] Exported successfully to -> {csv_target}")

if __name__ == '__main__':
    start_job()
