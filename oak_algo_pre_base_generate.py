"""
Complete Step Counting Pipeline - Standalone Version
1. Converts ActiGraph 80Hz data to 10Hz
2. Processes step counting
3. Generates individual results for each file
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from ssqueezepy import ssq_cwt


# ============================================================================
# STEP 1: DATA CONVERSION (80Hz -> 10Hz)
# ============================================================================

def convert_actigraph_to_10hz(filename: str, use_averaging: bool = False) -> pd.DataFrame:
    """Convert ActiGraph data from 80Hz to 10Hz."""
    with open(filename) as f:
        lines = f.readlines()

    start_date = ""
    start_time = ""
    sample_rate = 80

    for i, line in enumerate(lines):
        if line.startswith("Start Time"):
            start_time = line.split()[-1].strip()
        elif line.startswith("Start Date"):
            start_date = line.split()[-1].strip()
        elif line.startswith("Sample Rate"):
            sample_rate = int(line.split()[-1].strip())
        if line.startswith("Accelerometer X"):
            data_start_line = i
            break

    start = datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M:%S")
    data = pd.read_csv(filename, skiprows=data_start_line, header=0)

    if use_averaging:
        # Averaging method
        downsample_factor = sample_rate // 10
        num_full_windows = len(data) // downsample_factor
        data_trimmed = data.iloc[:num_full_windows * downsample_factor]
        
        x_avg = data_trimmed['Accelerometer X'].values.reshape(-1, downsample_factor).mean(axis=1)
        y_avg = data_trimmed['Accelerometer Y'].values.reshape(-1, downsample_factor).mean(axis=1)
        z_avg = data_trimmed['Accelerometer Z'].values.reshape(-1, downsample_factor).mean(axis=1)
        
        num_samples = len(x_avg)
        start_ms = int(start.timestamp() * 1000)
        timestamps_ms = [start_ms + int(i * 100) for i in range(num_samples)]
        
        df_formatted = pd.DataFrame({
            'timestamp': timestamps_ms,
            'x': x_avg,
            'y': y_avg,
            'z': z_avg
        })
    else:
        # Decimation method
        step_ms = 1000 / sample_rate
        num_samples = len(data)
        start_ms = int(start.timestamp() * 1000)
        timestamps_ms = [start_ms + int(i * step_ms) for i in range(num_samples)]
        
        df_original = pd.DataFrame({
            'timestamp': timestamps_ms,
            'x': data['Accelerometer X'].values,
            'y': data['Accelerometer Y'].values,
            'z': data['Accelerometer Z'].values
        })
        
        downsample_factor = sample_rate // 10
        df_formatted = df_original.iloc[::downsample_factor].copy().reset_index(drop=True)
    
    return df_formatted


# ============================================================================
# STEP 2: STEP COUNTING ALGORITHMS (from base.py)
# ============================================================================

def preprocess_bout(t_bout: np.ndarray, x_bout: np.ndarray, y_bout: np.ndarray,
                    z_bout: np.ndarray, fs: int = 10) -> tuple:
    """Preprocesses accelerometer bout to a common format."""
    if (len(t_bout) < 2 or len(x_bout) < 2 or
        len(y_bout) < 2 or len(z_bout) < 2):
        return np.array([]), np.array([])

    t_bout_interp = t_bout - t_bout[0]
    t_bout_interp = np.arange(t_bout_interp[0], t_bout_interp[-1], (1/fs))
    t_bout_interp = t_bout_interp + t_bout[0]

    f = interpolate.interp1d(t_bout, x_bout)
    x_bout_interp = f(t_bout_interp)

    f = interpolate.interp1d(t_bout, y_bout)
    y_bout_interp = f(t_bout_interp)

    f = interpolate.interp1d(t_bout, z_bout)
    z_bout_interp = f(t_bout_interp)

    x_bout_interp = adjust_bout(x_bout_interp)
    y_bout_interp = adjust_bout(y_bout_interp)
    z_bout_interp = adjust_bout(z_bout_interp)

    num_seconds = np.floor(len(x_bout_interp)/fs)

    t_bout_interp = t_bout_interp[:int(num_seconds*fs)]
    t_bout_interp = t_bout_interp[::fs]

    vm_bout_interp = np.sqrt(x_bout_interp**2 + y_bout_interp**2 + z_bout_interp**2)

    if vm_bout_interp.shape[0] > 0 and np.mean(vm_bout_interp) > 5:
        x_bout_interp = x_bout_interp/9.80665
        y_bout_interp = y_bout_interp/9.80665
        z_bout_interp = z_bout_interp/9.80665

    vm_bout_interp = np.sqrt(x_bout_interp**2 + y_bout_interp**2 + z_bout_interp**2) - 1

    return t_bout_interp, vm_bout_interp


def adjust_bout(inarray: np.ndarray, fs: int = 10) -> np.ndarray:
    """Fills observations in incomplete bouts."""
    if len(inarray) % fs >= 0.7*fs:
        for i in range(fs-len(inarray) % fs):
            inarray = np.append(inarray, inarray[-1])
    else:
        inarray = inarray[np.arange(len(inarray)//fs*fs)]
    return inarray


def get_pp(vm_bout: np.ndarray, fs: int = 10) -> npt.NDArray[np.float64]:
    """Calculate peak-to-peak metric in one-second time windows."""
    vm_res_sec = vm_bout.reshape((fs, -1), order="F")
    pp = np.ptp(vm_res_sec, axis=0)
    return pp


def compute_interpolate_cwt(tapered_bout: np.ndarray, fs: int = 10,
                            wavelet: tuple = ('gmw', {'beta': 90, 'gamma': 3})) -> tuple:
    """Compute and interpolate CWT over acceleration data."""
    window = tukey(len(tapered_bout), alpha=0.02, sym=True)
    tapered_bout = np.concatenate((np.zeros(5*fs), tapered_bout*window, np.zeros(5*fs)))

    out = ssq_cwt(tapered_bout[:-1], wavelet, fs=10)
    coefs = out[0]
    coefs = np.append(coefs, coefs[:, -1:], 1)
    coefs = coefs.astype('complex128')

    coefs = np.abs(coefs**2)

    freqs = out[2]
    freqs_interp = np.arange(0.5, 4.5, 0.05)
    interpolator = interpolate.RegularGridInterpolator((freqs, range(coefs.shape[1])), coefs)
    grid_x, grid_y = np.meshgrid(freqs_interp, range(coefs.shape[1]), indexing='ij')
    coefs_interp = interpolator((grid_x, grid_y))

    coefs_interp = coefs_interp[:, 5*fs:-5*fs]

    return freqs_interp, coefs_interp


def identify_peaks_in_cwt(freqs_interp: np.ndarray, coefs_interp: np.ndarray,
                          fs: int = 10, step_freq: tuple = (1.4, 2.3),
                          alpha: float = 0.6, beta: float = 2.5):
    """Identify dominant peaks in wavelet coefficients."""
    num_rows, num_cols = coefs_interp.shape
    num_cols2 = int(num_cols/fs)

    dp = np.zeros((num_rows, num_cols2))

    loc_min = np.argmin(abs(freqs_interp-step_freq[0]))
    loc_max = np.argmin(abs(freqs_interp-step_freq[1]))

    for i in range(num_cols2):
        x_start = i*fs
        x_end = (i + 1)*fs

        window = np.sum(coefs_interp[:, np.arange(x_start, x_end)], axis=1)

        locs, _ = find_peaks(window)
        pks = window[locs]
        ind = np.argsort(-pks)

        locs = locs[ind]
        pks = pks[ind]

        index_in_range = None

        for j, locs_j in enumerate(locs):
            if loc_min <= locs_j <= loc_max:
                index_in_range = j
                break

        peak_vec = np.zeros(num_rows)

        if index_in_range is not None:
            peaks_below = pks[locs < loc_min]
            peaks_above = pks[locs > loc_max]

            max_peak_magnitude_a = np.max(peaks_below) if len(peaks_below) > 0 else 0
            max_peak_magnitude_b = np.max(peaks_above) if len(peaks_above) > 0 else 0

            if (max_peak_magnitude_b / pks[index_in_range] < beta or 
                max_peak_magnitude_a / pks[index_in_range] < alpha):
                peak_vec[locs[index_in_range]] = 1
        dp[:, i] = peak_vec

    return dp


def find_continuous_dominant_peaks(valid_peaks: np.ndarray, min_t: int,
                                   delta: int) -> npt.NDArray[np.float64]:
    """Identifies continuous and sustained peaks within matrix."""
    num_rows, num_cols = valid_peaks.shape

    extended_peaks = np.zeros((num_rows, num_cols + 1), dtype=valid_peaks.dtype)
    extended_peaks[:, :num_cols] = valid_peaks

    cont_peaks = np.zeros_like(extended_peaks)

    for slice_ind in range(num_cols + 1 - min_t):
        slice_mat = extended_peaks[:, slice_ind:slice_ind + min_t]

        windows = list(range(min_t)) + list(range(min_t-2, -1, -1))
        stop = True

        for win_ind in windows:
            pr = np.where(slice_mat[:, win_ind] != 0)[0]
            stop = True

            for p in pr:
                index = np.arange(max(0, p - delta), min(p + delta + 1, num_rows))

                peaks1 = slice_mat[p, win_ind]
                peaks2 = peaks1
                if win_ind == 0:
                    peaks1 += slice_mat[index, win_ind + 1]
                elif win_ind == min_t - 1:
                    peaks1 += slice_mat[index, win_ind - 1]
                else:
                    peaks1 += slice_mat[index, win_ind - 1]
                    peaks2 += slice_mat[index, win_ind + 1]

                if win_ind == 0 or win_ind == min_t - 1:
                    if np.any(peaks1 > 1):
                        stop = False
                    else:
                        slice_mat[p, win_ind] = 0
                else:
                    if np.any(peaks1 > 1) and np.any(peaks2 > 1):
                        stop = False
                    else:
                        slice_mat[p, win_ind] = 0

            if stop:
                break

        if not stop:
            cont_peaks[:, slice_ind:slice_ind + min_t] = slice_mat

    return cont_peaks[:, :-1]


def find_walking(vm_bout: np.ndarray, fs: int = 10, min_amp: float = 0.3,
                 step_freq: tuple = (1.4, 2.3), alpha: float = 0.6,
                 beta: float = 2.5, min_t: int = 3,
                 delta: int = 20) -> npt.NDArray[np.float64]:
    """Finds walking and calculate steps from raw acceleration data."""
    wavelet = ('gmw', {'beta': 90, 'gamma': 3})

    pp = get_pp(vm_bout, fs)

    valid = np.ones(len(pp), dtype=bool)

    valid[pp < min_amp] = False

    if sum(valid) >= min_t:
        tapered_bout = vm_bout[np.repeat(valid, fs)]

        freqs_interp, coefs_interp = compute_interpolate_cwt(tapered_bout, fs, wavelet)

        dp = identify_peaks_in_cwt(freqs_interp, coefs_interp, fs, step_freq, alpha, beta)

        valid_peaks = np.zeros((dp.shape[0], len(valid)))
        valid_peaks[:, valid] = dp

        cont_peaks = find_continuous_dominant_peaks(valid_peaks, min_t, delta)

        cad = np.zeros(valid_peaks.shape[1])
        for i in range(len(cad)):
            ind_freqs = np.where(cont_peaks[:, i] > 0)[0]
            if len(ind_freqs) > 0:
                cad[i] = freqs_interp[ind_freqs[0]]
    else:
        cad = np.zeros(int(vm_bout.shape[0]/fs))

    return cad


# ============================================================================
# STEP 3: COMPLETE PIPELINE
# ============================================================================

def process_single_file(filepath, output_dir, use_averaging=False):
    """Complete pipeline: convert and process single file."""
    
    # Convert 80Hz to 10Hz
    data = convert_actigraph_to_10hz(filepath, use_averaging)
    
    # Extract arrays
    timestamp = np.array(data["timestamp"]) / 1000.0  # Convert to seconds
    x = np.array(data["x"], dtype="float64")
    y = np.array(data["y"], dtype="float64")
    z = np.array(data["z"], dtype="float64")
    
    # Preprocess
    t_bout_interp, vm_bout = preprocess_bout(timestamp, x, y, z, fs=10)
    
    if len(t_bout_interp) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Find walking
    cadence_bout = find_walking(vm_bout, fs=10, min_amp=0.3, step_freq=(1.4, 2.3),
                                alpha=0.6, beta=2.5, min_t=3, delta=20)
    
    # Create results
    results = pd.DataFrame({
        'timestamp': t_bout_interp,
        'datetime': pd.to_datetime(t_bout_interp, unit='s'),
        'cadence_hz': cadence_bout,
        'steps_per_minute': cadence_bout * 60,
        'is_walking': (cadence_bout > 0).astype(int)
    })
    
    # Summary
    total_steps = int(np.sum(cadence_bout))
    walking_seconds = int(np.sum(cadence_bout > 0))
    walking_minutes = walking_seconds / 60
    avg_cadence_hz = np.mean(cadence_bout[cadence_bout > 0]) if walking_seconds > 0 else 0
    avg_cadence_spm = avg_cadence_hz * 60
    
    summary = pd.DataFrame({
        'metric': ['total_steps', 'walking_time_seconds', 'walking_time_minutes',
                   'average_cadence_steps_per_minute', 'total_duration_minutes'],
        'value': [total_steps, walking_seconds, walking_minutes, avg_cadence_spm,
                 (timestamp[-1] - timestamp[0]) / 60]
    })
    
    # Save to individual folder
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join(output_dir, base_filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    results.to_csv(os.path.join(file_output_dir, f"{base_filename}_detailed_results.csv"), index=False)
    summary.to_csv(os.path.join(file_output_dir, f"{base_filename}_summary.csv"), index=False)
    
    return results, summary


def process_multiple_files(input_files, output_directory, use_averaging=False):
    """Process multiple files through complete pipeline."""
    print(f"\nProcessing {len(input_files)} file(s)...\n")
    
    results_summary = []
    
    for i, filepath in enumerate(input_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{i}/{len(input_files)}] {filename}...", end=" ")
        
        try:
            results_df, summary_df = process_single_file(filepath, output_directory, use_averaging)
            
            if len(results_df) > 0:
                total_steps = summary_df[summary_df['metric'] == 'total_steps']['value'].values[0]
                walking_time = summary_df[summary_df['metric'] == 'walking_time_minutes']['value'].values[0]
                
                results_summary.append({
                    'filename': filename,
                    'total_steps': int(total_steps),
                    'walking_time_minutes': round(walking_time, 2),
                    'status': 'Success'
                })
                print(f"✓ {int(total_steps)} steps")
            else:
                results_summary.append({
                    'filename': filename,
                    'total_steps': 0,
                    'walking_time_minutes': 0,
                    'status': 'No valid data'
                })
                print("⚠ No data")
        except Exception as e:
            results_summary.append({
                'filename': filename,
                'total_steps': 0,
                'walking_time_minutes': 0,
                'status': f'Error: {str(e)}'
            })
            print(f"✗ Error")
    
    # Save summary
    pd.DataFrame(results_summary).to_csv(
        os.path.join(output_directory, "_ALL_FILES_SUMMARY.csv"), index=False
    )
    
    success_count = sum(1 for r in results_summary if r['status'] == 'Success')
    print(f"\nComplete: {success_count}/{len(input_files)} successful")


def main():
    root = tk.Tk()
    root.withdraw()
    
    # Select input directory
    input_directory = filedialog.askdirectory(
        title="Select Input Directory (raw ActiGraph CSV files)"
    )
    if not input_directory:
        return
    
    input_files = [os.path.join(input_directory, f) 
                   for f in os.listdir(input_directory) if f.endswith('.csv')]
    
    if not input_files:
        print("No CSV files found")
        return
    
    print(f"\nFound {len(input_files)} CSV file(s)")
    
    # Select output directory
    output_directory = filedialog.askdirectory(
        title="Select Output Directory (for results)"
    )
    if not output_directory:
        return
    
    # Ask for downsampling method
    use_averaging = messagebox.askyesno(
        "Downsampling Method",
        "Use averaging for 80Hz->10Hz conversion?\n\n"
        "YES: Average values (smoother)\n"
        "NO: Take every 8th sample (faster)"
    )
    
    # Process all files
    process_multiple_files(input_files, output_directory, use_averaging)


if __name__ == "__main__":
    main()