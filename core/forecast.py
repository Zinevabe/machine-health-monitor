import numpy as np
import datetime

def calc_breakdown_point(hist_data, danger_mark=7.1):
    if len(hist_data) < 2:
        return "Stable or Improving", None

    # Time values as days
    time_series_x = [t.toordinal() for t in hist_data["Timestamp"]]
    y_vibes = hist_data["RMS_Value"].tolist()

    try:
        # Fit a 1-degree polynomial (straight line)
        m_curve, b_intercept = np.polyfit(time_series_x, y_vibes, 1)
    except Exception:
        return "Stable or Improving", None

    if m_curve <= 0.0:
        return "Stable or Improving", None

    # Calculate when it hits danger_mark
    intersection_x = (danger_mark - b_intercept) / m_curve
    
    try:
        crash_dt = datetime.date.fromordinal(int(intersection_x))
        return "Degrading", crash_dt
    except ValueError:
        return "Degrading", None
