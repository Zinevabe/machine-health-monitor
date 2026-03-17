import os

def assign_vibration_class(amplitude):
    limit_y = float(os.environ.get("LVL_WARN", "2.3"))
    limit_o = float(os.environ.get("LVL_ALERT", "4.5"))
    limit_r = float(os.environ.get("LVL_CRIT", "7.1"))

    if amplitude <= limit_y:
        return "Zone A (Green) - Newly commissioned"
    elif amplitude <= limit_o:
        return "Zone B (Yellow) - Unrestricted operation"
    elif amplitude <= limit_r:
        return "Zone C (Orange) - Restricted operation"
    
    return "Zone D (Red) - DAMAGE OCCURS"

def attach_health_zones(dataset):
    dataset["ISO_Status"] = [assign_vibration_class(v) for v in dataset["Velocity_RMS"]]
    return dataset
