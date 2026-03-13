import os
import pandas as pd
import numpy as np
from pathlib import Path

def get_rms_math(arr):
    if not arr:
        return 0.0
    return float(np.sqrt(np.mean(np.array(arr)**2)))

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
        raw_amplitudes = []
        is_reading_data = False
        
        for q, row_str in enumerate(lines_data):
            row_clean = row_str.strip()
            
            # If we hit a new Equipment block, save the previous one if it exists
            if row_clean.startswith("Equipment:"):
                if is_reading_data and machine != "UNKNOWN" and raw_amplitudes:
                    # Clean up machine name inconsistencies (e.g. remove CHPP tag)
                    clean_machine_name = machine.replace("(CHPP)", "").strip()
                    extracted_items.append({
                        "Equipment": clean_machine_name,
                        "Timestamp": recorded_time,
                        "RMS_Value": get_rms_math(raw_amplitudes),
                        "Source_File": txt_file.name
                    })
                
                machine = row_clean.split(":", 1)[1].strip()
                raw_amplitudes = [] # Reset for new block
                is_reading_data = False
                
            elif row_clean.startswith("Date/Time:"):
                dt_part = row_clean.replace("Date/Time:", "").split("Amplitude")[0].strip()
                recorded_time = pd.to_datetime(dt_part)
                
            elif row_clean.startswith("---------"):
                is_reading_data = True
                
            elif is_reading_data and row_clean:
                chunks = row_clean.replace(",", "").split()
                for i in range(0, len(chunks)-1, 2):
                    amp_str = chunks[i+1]
                    
                    for sign in ["-", "+"]:
                        idx = amp_str.find(sign, 1)
                        if idx != -1 and "e" not in amp_str.lower():
                            amp_str = amp_str[:idx] + "e" + amp_str[idx:]
                            break
                    
                    try:
                        raw_amplitudes.append(float(amp_str))
                    except ValueError:
                        pass
        
        # Save the last block in the file
        if is_reading_data and machine != "UNKNOWN" and raw_amplitudes:
            clean_machine_name = machine.replace("(CHPP)", "").strip()
            extracted_items.append({
                "Equipment": clean_machine_name,
                "Timestamp": recorded_time,
                "RMS_Value": get_rms_math(raw_amplitudes),
                "Source_File": txt_file.name
            })
        
    res_df = pd.DataFrame(extracted_items)
    if not res_df.empty:
        res_df.sort_values("Timestamp", inplace=True)
        res_df.reset_index(drop=True, inplace=True)
        
    return res_df
