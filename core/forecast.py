import numpy as np
import pandas as pd
import datetime

def calc_breakdown_point(hist_data, danger_mark=7.1):
    if len(hist_data) < 2:
        return "Stable or Improving", None

    # Calculate days (as float) from the first record to avoid large X coordinates 
    # and handle multiple entries on the same day properly.
    start_date = hist_data["Timestamp"].iloc[0]
    time_series_x = [(t - start_date).total_seconds() / 86400.0 for t in hist_data["Timestamp"]]
    y_vibes = hist_data["RMS_Value"].tolist()

    # Prevent polyfit error if all data points are taken at the exact same exact timestamp
    if max(time_series_x) - min(time_series_x) == 0:
         return "Stable or Improving", None

    try:
        # Fit a 1-degree polynomial (y = mx + b)
        m_curve, b_intercept = np.polyfit(time_series_x, y_vibes, 1)
    except Exception:
        return "Stable or Improving", None

    # If the trend is going down or flat, it's not degrading
    if m_curve <= 0.0:
        return "Stable or Improving", None

    # Calculate how many days from the start_date it takes to hit danger_mark
    days_to_crash = (danger_mark - b_intercept) / m_curve
    
    # If the calculated days point to the past, the trend doesn't make sense
    if days_to_crash < 0:
        return "Degrading", None
        
    try:
        crash_dt = start_date + pd.Timedelta(days=int(days_to_crash))
        return "Degrading", crash_dt.strftime('%Y-%m-%d')
    except (ValueError, OverflowError):
        return "Degrading", None
