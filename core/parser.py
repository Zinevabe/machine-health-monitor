import os
import pandas as pd
import numpy as np
from pathlib import Path

def get_rms_math(times_ms, amplitudes_g):
    if not amplitudes_g or len(times_ms) != len(amplitudes_g) or len(times_ms) < 2:
        return 0.0
        
    # 0. VERY IMPORTANT: Sort data by Time
    # Raw log reads left-to-right but columns are interleaved block-by-block.
    # Without sorting, Time jumps randomly (e.g., 0 -> 133 -> 266 -> 0.13) causing integration explosion!
    sort_idx = np.argsort(times_ms)
    t_sec = np.array(times_ms)[sort_idx] / 1000.0
    a_mms2 = np.array(amplitudes_g)[sort_idx] * 9806.65
    
    # 1. Detrend DC Offset from Acceleration
    a_mms2 = a_mms2 - np.mean(a_mms2)
    
    # 2. Integrate Acceleration to Velocity using Trapezoidal method
    v_mms = np.zeros_like(a_mms2)
    dt = np.diff(t_sec)
    a_avg = (a_mms2[:-1] + a_mms2[1:]) / 2.0
    v_mms[1:] = np.cumsum(a_avg * dt)
    
    # 3. Detrend DC Offset from Velocity (Remove drift)
    v_mms = v_mms - np.mean(v_mms)
    
    # 4. Calculate RMS of Velocity (mm/s)
    return float(np.sqrt(np.mean(v_mms**2)))

def collect_measurements(scan_dir):
    loc_path = Path(scan_dir)
    if not loc_path.is_dir():
        return pd.DataFrame()
        
    extracted_items = []
    
    for txt_file in loc_path.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines_data = f.readlines()
            
        machine = "UNKNOWN"
        recorded_time = None
        raw_times = []
        raw_amplitudes = []
        is_reading_data = False
        
        for q, row_str in enumerate(lines_data):
            row_clean = row_str.strip()
            
            # If we hit a new Equipment block, save the previous one if it exists
            if row_clean.startswith("Equipment:"):
                if is_reading_data and machine != "UNKNOWN" and raw_amplitudes:
                    clean_machine_name = machine.replace("(CHPP)", "").strip()
                    extracted_items.append({
                        "Equipment": clean_machine_name,
                        "Timestamp": recorded_time,
                        "RMS_Value": get_rms_math(raw_times, raw_amplitudes),
                        "Source_File": txt_file.name
                    })
                
                machine = row_clean.split(":", 1)[1].strip()
                raw_times = [] 
                raw_amplitudes = [] 
                is_reading_data = False
                
            elif row_clean.startswith("Date/Time:"):
                dt_part = row_clean.replace("Date/Time:", "").split("Amplitude")[0].strip()
                recorded_time = pd.to_datetime(dt_part)
                
            elif row_clean.startswith("---------"):
                is_reading_data = True
                
            elif is_reading_data and row_clean:
                chunks = row_clean.replace(",", "").split()
                for i in range(0, len(chunks)-1, 2):
                    t_str = chunks[i]
                    amp_str = chunks[i+1]
                    
                    for sign in ["-", "+"]:
                        idx = amp_str.find(sign, 1)
                        if idx != -1 and "e" not in amp_str.lower():
                            amp_str = amp_str[:idx] + "e" + amp_str[idx:]
                            break
                            
                    # Some time values might also trigger the float bug or be empty
                    try:
                        parsed_t = float(t_str)
                        parsed_a = float(amp_str)
                        raw_times.append(parsed_t)
                        raw_amplitudes.append(parsed_a)
                    except ValueError:
                        pass
        
        # Save the last block in the file
        if is_reading_data and machine != "UNKNOWN" and raw_amplitudes:
            clean_machine_name = machine.replace("(CHPP)", "").strip()
            extracted_items.append({
                "Equipment": clean_machine_name,
                "Timestamp": recorded_time,
                "RMS_Value": get_rms_math(raw_times, raw_amplitudes),
                "Source_File": txt_file.name
            })
        
    res_df = pd.DataFrame(extracted_items)
    if not res_df.empty:
        res_df.sort_values("Timestamp", inplace=True)
        res_df.reset_index(drop=True, inplace=True)
        
    return res_df
