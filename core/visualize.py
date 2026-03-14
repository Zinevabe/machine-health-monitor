import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # Suppress warnings

def generate_visualizations(logs_table, out_dir):
    print("\n[-] Generating Data Visualizations (Enhanced Insights)...")
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Timeline (Scatter & Line Plot) - Auto scaled to see the actual trend
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid", context="paper")
    
    for asst, group_df in logs_table.groupby("Equipment"):
        ordered = group_df.sort_values("Timestamp")
        plt.plot(ordered['Timestamp'], ordered['RMS_Value'], 
                 marker='o', linestyle='-', linewidth=2.5, markersize=8, label=asst)
    
    # Add back ISO Limits as data is now properly integrated to Velocity (mm/s)
    plt.axhline(y=4.5, color='orange', linestyle='--', alpha=0.7, label='Zone C (Warning)')
    plt.axhline(y=7.1, color='red', linestyle='--', alpha=0.7, label='Zone D (Danger)')
    
    plt.title('Machine Vibration Trends (Velocity RMS)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Date / Time', fontsize=12)
    plt.ylabel('Velocity RMS (mm/s)', fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    timeline_path = os.path.join(out_dir, "plot_1_vibration_timeline.png")
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Shift in Vibration (First vs Latest) - Grouped Bar Chart
    # -------------------------------------------------------------
    # This chart tells us "Did the machine get worse since its first record?"
    plt.figure(figsize=(10, 6))
    
    first_records = logs_table.loc[logs_table.groupby("Equipment")['Timestamp'].idxmin()][['Equipment', 'RMS_Value']]
    latest_records = logs_table.loc[logs_table.groupby("Equipment")['Timestamp'].idxmax()][['Equipment', 'RMS_Value']]
    
    first_records.rename(columns={'RMS_Value': 'Initial_RMS'}, inplace=True)
    latest_records.rename(columns={'RMS_Value': 'Latest_RMS'}, inplace=True)
    
    merged_shift = pd.merge(first_records, latest_records, on='Equipment')
    merged_shift_melted = merged_shift.melt(id_vars='Equipment', var_name='Reading', value_name='Velocity RMS (mm/s)')
    
    sns.barplot(data=merged_shift_melted, x='Velocity RMS (mm/s)', y='Equipment', hue='Reading', palette=['#3498db', '#e74c3c'])
    
    plt.title('Vibration Shift: Initial Calibration vs Latest Reading', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Velocity RMS (mm/s)', fontsize=12)
    plt.ylabel('Asset Name', fontsize=12)
    plt.legend(title='Record Time')
    plt.tight_layout()
    
    bar_path = os.path.join(out_dir, "plot_2_initial_vs_latest.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 3. Percentage Growth Heatmap (% Change from Baseline)
    # -------------------------------------------------------------
    # A raw value heatmap doesn't show much if values are similar. % Growth shows true degradation.
    plt.figure(figsize=(10, 6))
    
    logs_table['Month_Year'] = logs_table['Timestamp'].dt.strftime('%Y-%m')
    pivot_raw = logs_table.pivot_table(index='Equipment', columns='Month_Year', values='RMS_Value', aggfunc='mean')
    
    # Calculate percentage change based on the first recorded column
    first_col = pivot_raw.iloc[:, 0]
    growth_pivot = pivot_raw.apply(lambda x: ((x - first_col) / first_col) * 100)
    
    sns.heatmap(growth_pivot, annot=True, fmt=".1f", cmap="vlag", center=0,
                linewidths=1, linecolor='white', cbar_kws={'label': '% Change in Vibration'})
    
    plt.title('Degradation Heatmap (% Growth from Initial Measurement)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Recording Month', fontsize=12)
    plt.ylabel('Machine', fontsize=12)
    plt.tight_layout()
    
    heat_path = os.path.join(out_dir, "plot_3_growth_heatmap.png")
    plt.savefig(heat_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] 3 Enhanced Charts generated inside -> {out_dir}/")
