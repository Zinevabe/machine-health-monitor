import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Suppress tight_layout warnings

def generate_visualizations(logs_table, out_dir):
    """
    สร้างกราฟวิเคราะห์ข้อมูลแบบต่างๆ จาก Logs หมวดหมู่ 
    - กราฟ 1: Scatter & Line plot สำหรับดูแนวโน้ม (Time-Series Trend) แบบแยกเครื่อง
    - กราฟ 2: Bar chart เปรียบเทียบสถานะ RMS ล่าสุดเครื่องจักรทั้งหมด
    - กราฟ 3: Heatmap สรุปค่าสั่นแยกตามเวลา
    """
    print("\n[-] Generating Data Visualizations...")
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Timeline (Scatter & Line Plot) - การเติบโตของการสั่นตามเวลา
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid", context="paper")
    
    # Plot individual machine timelines
    for asst, group_df in logs_table.groupby("Equipment"):
        ordered = group_df.sort_values("Timestamp")
        plt.plot(ordered['Timestamp'], ordered['RMS_Value'], 
                 marker='o', linestyle='-', linewidth=2, markersize=8, label=asst)
    
    plt.axhline(y=4.5, color='orange', linestyle='--', alpha=0.7, label='Zone C (Warning)')
    plt.axhline(y=7.1, color='red', linestyle='--', alpha=0.7, label='Zone D (Danger)')

    plt.title('Machine Vibration Trends Over Time', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Date / Time', fontsize=12)
    plt.ylabel('Vibration RMS (G)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    timeline_path = os.path.join(out_dir, "plot_1_vibration_timeline.png")
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Latest Asset Condition (Bar Chart)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Extract just the latest record for each machine
    latest_idx = logs_table.groupby("Equipment")['Timestamp'].idxmax()
    latest_df = logs_table.loc[latest_idx].sort_values("RMS_Value", ascending=False)
    
    # Assign colors based on zone (A=green, B=yellow, C=orange, D=red)
    def determine_color(val):
        if val >= 7.1: return '#e74c3c'
        elif val >= 4.5: return '#f39c12'
        elif val >= 2.3: return '#f1c40f'
        return '#2ecc71'
        
    bar_colors = [determine_color(x) for x in latest_df['RMS_Value']]
    
    ax = sns.barplot(x="RMS_Value", y="Equipment", data=latest_df, palette=bar_colors)
    
    # Add value text at the end of each bar
    for index, value in enumerate(latest_df['RMS_Value']):
        ax.text(value + 0.02, index, f'{value:.2f}', va='center')

    plt.title('Current Machine Health Condition (Latest RMS)', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Vibration RMS (G)', fontsize=12)
    plt.ylabel('Asset Name', fontsize=12)
    plt.tight_layout()
    
    bar_path = os.path.join(out_dir, "plot_2_latest_condition_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 3. Vibration Heatmap (Month vs Equipment)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Create month-year column for grouping
    logs_table['Month_Year'] = logs_table['Timestamp'].dt.strftime('%Y-%m')
    
    # Pivot for Heatmap (take mean if multiple reads in same month)
    pivot_tb = logs_table.pivot_table(
        index='Equipment', 
        columns='Month_Year', 
        values='RMS_Value', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_tb, annot=True, fmt=".2f", cmap="YlOrRd", 
                linewidths=0.5, cbar_kws={'label': 'RMS (G)'})
    
    plt.title('Average Vibration Heatmap by Month', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Recording Month', fontsize=12)
    plt.ylabel('Machine', fontsize=12)
    plt.tight_layout()
    
    heat_path = os.path.join(out_dir, "plot_3_heat_map.png")
    plt.savefig(heat_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] 3 Charts generated inside -> {out_dir}/")
