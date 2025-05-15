import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

def find_peak(acc, peak_win_len=3):
    half_k = np.round(peak_win_len / 2).astype(int)  # Use int instead of np.int
    segments = np.floor(acc.shape[0] / peak_win_len).astype(int)  # Use int instead of np.int
    peak_info = np.empty((segments, 5))
    peak_info[:] = np.inf
    for i in np.arange(segments):
        start_idx = i * peak_win_len
        end_idx = start_idx + peak_win_len - 1
        tmp_loc_a = np.argmax(acc[start_idx:end_idx + 1])
        tmp_loc_b = i * peak_win_len + tmp_loc_a
        start_idx_ctr = max(0, tmp_loc_b - half_k)
        end_idx_ctr = min(len(acc), tmp_loc_b + half_k)
        
        check_loc = np.argmax(acc[start_idx_ctr:end_idx_ctr + 1])
        if check_loc == half_k:
            peak_info[i, 0] = tmp_loc_b
            peak_info[i, 1] = np.max(acc[start_idx:end_idx + 1])
    peak_info = peak_info[~np.isinf(peak_info[:, 0])]
    return peak_info

def filter_magnitude(peak_info, mag_thres=1.2):
    return peak_info[peak_info[:, 1] > mag_thres]

def calc_periodicity(peak_info, period_min=5, period_max=15):
    num_peaks = peak_info.shape[0]
    peak_info[:num_peaks - 1, 2] = np.diff(peak_info[:, 0])
    peak_info = peak_info[(peak_info[:, 2] > period_min) & (peak_info[:, 2] < period_max)]
    return peak_info

def clac_similarity(peak_info, sim_thres=-0.5):
    num_peaks = len(peak_info[:, 1])
    peak_info[:(num_peaks - 2), 3] = -np.abs(peak_info[:, 1][2:] - peak_info[:, 1][:-2])
    peak_info = peak_info[peak_info[:, 3] > sim_thres]
    return peak_info

def filter_by_continue_threshold_variance_threshold(peak_info, acc, cont_win_size=3, cont_thres=4, var_thres=0.001):
    end_for = len(peak_info[:, 2]) - 1
    for i in np.arange(cont_thres - 1, end_for):
        v_count = sum(np.var(acc[int(peak_info[i - x + 1, 0]):int(peak_info[i - x + 2, 0] + 1)], ddof=1) > var_thres 
                       for x in range(1, cont_thres + 1))
        peak_info[i, 4] = 1 if v_count >= cont_win_size else 0
    peak_info = peak_info[peak_info[:, 4] == 1, 0]
    return peak_info

def counts_peaks(peak_info, acc, fs=15):
    peak_info_count = np.ceil(peak_info / fs).astype(int).tolist()
    peak_info_count_dict = {x: peak_info_count.count(x) for x in set(peak_info_count)}
    all_steps = pd.Series(np.arange(np.floor(acc.shape[0] / fs).astype(int)))
    return all_steps.map(peak_info_count_dict).fillna(0)

def read_data(filename: str) -> pd.DataFrame:
    print(f"Reading data from file: {filename}")  # Debug output
    with open(filename) as f:
        lines = f.readlines()

    start_date = ""
    start_time = ""

    for line in lines:
        if line.startswith("Start Time"):
            start_time = line.split()[-1].strip().strip('"')  # Strip quotes
        elif line.startswith("Start Date"):
            start_date = line.split()[-1].strip().strip('"')  # Strip quotes
        if line.startswith("Accelerometer X"):
            break

    start = datetime.strptime(f"{start_date} {start_time}", "%d/%m/%Y %H:%M:%S")
    data = pd.read_csv(filename, skiprows=len(lines) - len(lines[lines.index(line):]), header=0)
    step = timedelta(seconds=1 / 80)
    timestamps = [start + i * step for i in range(len(data))]
    df_data = pd.DataFrame(data.values, columns=data.columns)
    df_data['dateTime'] = timestamps
    df_data = df_data[['dateTime'] + list(data.columns)]

    return df_data

def process_single_file(filepath: str):
    try:
        df = read_data(filepath)
        df.set_index('dateTime', inplace=True)
        df_resampled = df.resample('66.67ms').mean()
        df_resampled.reset_index(inplace=True)
        selected_columns_data = df_resampled.iloc[:, 1:4].values
        return selected_columns_data
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")

def step_counts_per_sec(raw_acc, peak_win_len=3, period_min=5, period_max=15, fs=15, mag_thres=1.2,
                        cont_win_size=3, cont_thres=4, var_thres=0.001):
    acc = np.sqrt(np.power(raw_acc[:, 0], 2) + np.power(raw_acc[:, 1], 2) + np.power(raw_acc[:, 2], 2))
    peak_data = find_peak(acc, peak_win_len)
    peak_data = filter_magnitude(peak_data, mag_thres)
    peak_data = calc_periodicity(peak_data, period_min, period_max)
    peak_data = clac_similarity(peak_data)
    peak_data = filter_by_continue_threshold_variance_threshold(peak_data, acc, cont_win_size, cont_thres, var_thres)
    peak_data = counts_peaks(peak_data, acc, fs)
    return peak_data

# Entry point for script execution
if __name__ == "__main__":
    if len(sys.argv) != 2:  # Ensure expected argument count
        print("Usage: python verisense_resample_datetime.py <address_text_file>")
        sys.exit(1)

    address_text_file = sys.argv[1]  # First argument
    if not os.path.isfile(address_text_file):
        print("The provided address file does not exist. Exiting.")
        sys.exit(1)

    with open(address_text_file, 'r') as f:
        input_file = f.readline().strip().strip('"')  # Read the first line only
    if not input_file:
        print("No valid input file found in the provided address file. Exiting.")
        sys.exit(1)

    raw_acc = process_single_file(input_file)

    if raw_acc is not None and len(raw_acc) > 0:
        steps = step_counts_per_sec(raw_acc)
        print(f"Total steps are: {sum(steps)}")
    else:
        print("No acceleration data available.")