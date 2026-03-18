import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis

G_TO_MMS2 = 9806.65


def _safe_rms(values):
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values ** 2)))


def _prepare_waveform(times_ms, amplitudes_g):
    if not amplitudes_g or len(times_ms) != len(amplitudes_g) or len(times_ms) < 2:
        return None

    sort_idx = np.argsort(times_ms)
    t_sec = np.asarray(times_ms, dtype=float)[sort_idx] / 1000.0
    accel_g = np.asarray(amplitudes_g, dtype=float)[sort_idx]

    valid_mask = np.isfinite(t_sec) & np.isfinite(accel_g)
    t_sec = t_sec[valid_mask]
    accel_g = accel_g[valid_mask]
    if t_sec.size < 2:
        return None

    accel_g_detrended = accel_g - np.mean(accel_g)
    accel_mms2 = accel_g_detrended * G_TO_MMS2

    velocity_mms = np.zeros_like(accel_mms2)
    dt = np.diff(t_sec)
    accel_avg = (accel_mms2[:-1] + accel_mms2[1:]) / 2.0
    velocity_mms[1:] = np.cumsum(accel_avg * dt)
    velocity_mms = velocity_mms - np.mean(velocity_mms)

    return t_sec, accel_g_detrended, velocity_mms


def _extract_frequency_features(t_sec, signal_values):
    feature_defaults = {
        "Sample_Rate_Hz": 0.0,
        "Dominant_Frequency_Hz": 0.0,
        "Dominant_Frequency_Amplitude": 0.0,
        "Spectral_Centroid_Hz": 0.0,
        "Spectral_Entropy": 0.0,
    }

    if signal_values.size < 4:
        return feature_defaults

    dt = np.diff(t_sec)
    valid_dt = dt[dt > 0]
    if valid_dt.size == 0:
        return feature_defaults

    sample_interval = float(np.median(valid_dt))
    if sample_interval <= 0:
        return feature_defaults

    sample_rate_hz = 1.0 / sample_interval
    centered_signal = signal_values - np.mean(signal_values)
    fft_values = np.fft.rfft(centered_signal)
    freqs = np.fft.rfftfreq(centered_signal.size, d=sample_interval)

    if freqs.size <= 1:
        feature_defaults["Sample_Rate_Hz"] = sample_rate_hz
        return feature_defaults

    freqs = freqs[1:]
    spectrum = np.abs(fft_values[1:])
    power = spectrum ** 2

    if np.allclose(power.sum(), 0.0):
        feature_defaults["Sample_Rate_Hz"] = sample_rate_hz
        return feature_defaults

    dominant_idx = int(np.argmax(power))
    spectral_prob = power / power.sum()
    spectral_entropy = 0.0
    if spectral_prob.size > 1:
        spectral_entropy = float(
            -np.sum(spectral_prob * np.log2(spectral_prob + 1e-12)) / np.log2(spectral_prob.size)
        )

    return {
        "Sample_Rate_Hz": float(sample_rate_hz),
        "Dominant_Frequency_Hz": float(freqs[dominant_idx]),
        "Dominant_Frequency_Amplitude": float(spectrum[dominant_idx] / centered_signal.size),
        "Spectral_Centroid_Hz": float(np.sum(freqs * power) / power.sum()),
        "Spectral_Entropy": spectral_entropy,
    }


def extract_measurement_features(times_ms, amplitudes_g):
    waveform = _prepare_waveform(times_ms, amplitudes_g)
    feature_defaults = {
        "Sample_Count": 0,
        "Duration_ms": 0.0,
        "Acceleration_RMS_G": 0.0,
        "Peak_Acceleration_G": 0.0,
        "Peak_to_Peak_G": 0.0,
        "Kurtosis": 0.0,
        "Crest_Factor": 0.0,
        "Velocity_RMS": 0.0,
        "Velocity_Peak_mm_s": 0.0,
        "Sample_Rate_Hz": 0.0,
        "Dominant_Frequency_Hz": 0.0,
        "Dominant_Frequency_Amplitude": 0.0,
        "Spectral_Centroid_Hz": 0.0,
        "Spectral_Entropy": 0.0,
    }

    if waveform is None:
        return feature_defaults

    t_sec, accel_g_detrended, velocity_mms = waveform
    accel_rms = _safe_rms(accel_g_detrended)
    peak_accel = float(np.max(np.abs(accel_g_detrended)))

    features = {
        "Sample_Count": int(accel_g_detrended.size),
        "Duration_ms": float((t_sec[-1] - t_sec[0]) * 1000.0),
        "Acceleration_RMS_G": accel_rms,
        "Peak_Acceleration_G": peak_accel,
        "Peak_to_Peak_G": float(np.ptp(accel_g_detrended)),
        "Kurtosis": float(kurtosis(accel_g_detrended, fisher=False, bias=False))
        if accel_g_detrended.size > 3
        else 0.0,
        "Crest_Factor": float(peak_accel / accel_rms) if accel_rms > 0 else 0.0,
        "Velocity_RMS": _safe_rms(velocity_mms),
        "Velocity_Peak_mm_s": float(np.max(np.abs(velocity_mms))) if velocity_mms.size else 0.0,
    }
    features.update(_extract_frequency_features(t_sec, accel_g_detrended))
    return features


def collect_measurements(scan_dir):
    loc_path = Path(scan_dir)
    if not loc_path.is_dir():
        return pd.DataFrame()

    extracted_items = []

    for txt_file in sorted(loc_path.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as file_handle:
            lines_data = file_handle.readlines()

        machine = "UNKNOWN"
        recorded_time = None
        raw_times = []
        raw_amplitudes = []
        is_reading_data = False

        def flush_measurement():
            if not is_reading_data or machine == "UNKNOWN" or not raw_amplitudes:
                return

            clean_machine_name = machine.replace("(CHPP)", "").strip()
            extracted_items.append(
                {
                    "Equipment": clean_machine_name,
                    "Timestamp": recorded_time,
                    "Source_File": txt_file.name,
                    **extract_measurement_features(raw_times, raw_amplitudes),
                }
            )

        for row_str in lines_data:
            row_clean = row_str.strip()

            if row_clean.startswith("Equipment:"):
                flush_measurement()
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
                for idx in range(0, len(chunks) - 1, 2):
                    t_str = chunks[idx]
                    amp_str = chunks[idx + 1]

                    for sign in ["-", "+"]:
                        sign_idx = amp_str.find(sign, 1)
                        if sign_idx != -1 and "e" not in amp_str.lower():
                            amp_str = amp_str[:sign_idx] + "e" + amp_str[sign_idx:]
                            break

                    try:
                        raw_times.append(float(t_str))
                        raw_amplitudes.append(float(amp_str))
                    except ValueError:
                        continue

        flush_measurement()

    res_df = pd.DataFrame(extracted_items)
    if not res_df.empty:
        res_df.sort_values(["Equipment", "Timestamp"], inplace=True)
        res_df.reset_index(drop=True, inplace=True)

    return res_df
