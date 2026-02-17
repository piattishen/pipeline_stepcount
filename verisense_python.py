"""
Unified Verisense Step Counting Algorithm
Processes ActiGraph accelerometer data and counts steps
Outputs per-file results with START_TIME, STOP_TIME, STEP format
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import argparse


class VerisenseStepCounter:
    """
    Implementation of the Verisense step counting algorithm
    Based on Gu et al. 2017 method with additional magnitude threshold
    """
    
    def __init__(self, peak_win_len=3, period_min=5, period_max=15, 
                 sim_thres=-0.5, cont_win_size=3, cont_thres=4, 
                 var_thres=0.001, mag_thres=1.2, fs=15):
        """
        Initialize step counter with algorithm parameters
        
        Parameters:
        -----------
        peak_win_len : int
            Window length for peak detection (default: 3)
        period_min : int
            Minimum samples between steps (default: 5)
        period_max : int
            Maximum samples between steps (default: 15)
        sim_thres : float
            Similarity threshold (default: -0.5)
        cont_win_size : int
            Continuity window size (default: 3)
        cont_thres : int
            Continuity threshold (default: 4)
        var_thres : float
            Variance threshold (default: 0.001)
        mag_thres : float
            Magnitude threshold in g (default: 1.2)
        fs : int
            Sampling frequency in Hz (default: 15)
        """
        self.peak_win_len = peak_win_len
        self.period_min = period_min
        self.period_max = period_max
        self.sim_thres = sim_thres
        self.cont_win_size = cont_win_size
        self.cont_thres = cont_thres
        self.var_thres = var_thres
        self.mag_thres = mag_thres
        self.fs = fs
    
    def find_peak(self, acc):
        """
        Find peaks in acceleration signal
        
        Returns:
        --------
        peak_info : numpy array
            Matrix with columns: [location, magnitude, periodicity, similarity, continuity]
        """
        half_k = int(np.round(self.peak_win_len / 2))
        segments = int(np.floor(acc.shape[0] / self.peak_win_len))
        peak_info = np.empty((segments, 5))
        peak_info[:] = np.inf
        
        for i in range(segments):
            start_idx = i * self.peak_win_len
            end_idx = start_idx + self.peak_win_len - 1
            tmp_loc_a = np.argmax(acc[start_idx:end_idx + 1])
            tmp_loc_b = i * self.peak_win_len + tmp_loc_a
            
            start_idx_ctr = max(0, tmp_loc_b - half_k)
            end_idx_ctr = min(len(acc) - 1, tmp_loc_b + half_k)
            
            check_loc = np.argmax(acc[start_idx_ctr:end_idx_ctr + 1])
            if check_loc == half_k:
                peak_info[i, 0] = tmp_loc_b
                peak_info[i, 1] = np.max(acc[start_idx:end_idx + 1])
        
        # Remove rows with inf values
        peak_info = peak_info[~np.isinf(peak_info[:, 0])]
        return peak_info
    
    def filter_magnitude(self, peak_info):
        """Filter peaks by magnitude threshold"""
        return peak_info[peak_info[:, 1] > self.mag_thres]
    
    def calc_periodicity(self, peak_info):
        """Calculate and filter by periodicity"""
        if len(peak_info) == 0:
            return peak_info
        
        num_peaks = peak_info.shape[0]
        if num_peaks > 1:
            peak_info[:num_peaks - 1, 2] = np.diff(peak_info[:, 0])
            peak_info = peak_info[(peak_info[:, 2] > self.period_min) & 
                                 (peak_info[:, 2] < self.period_max)]
        return peak_info
    
    def calc_similarity(self, peak_info):
        """Calculate and filter by similarity"""
        if len(peak_info) < 3:
            return peak_info
        
        num_peaks = len(peak_info[:, 1])
        peak_info[:(num_peaks - 2), 3] = -np.abs(peak_info[:, 1][2:] - peak_info[:, 1][:-2])
        peak_info = peak_info[peak_info[:, 3] > self.sim_thres]
        return peak_info
    
    def filter_continuity(self, peak_info, acc):
        """Filter by continuity and variance thresholds"""
        if len(peak_info) < self.cont_thres:
            return np.array([])
        
        end_for = len(peak_info[:, 2]) - 1
        for i in range(self.cont_thres - 1, end_for):
            v_count = sum(
                np.var(acc[int(peak_info[i - x + 1, 0]):int(peak_info[i - x + 2, 0] + 1)], ddof=1) > self.var_thres 
                for x in range(1, self.cont_thres + 1)
            )
            peak_info[i, 4] = 1 if v_count >= self.cont_win_size else 0
        
        peak_info = peak_info[peak_info[:, 4] == 1, 0]
        return peak_info
    
    def count_steps_per_second(self, peak_info, acc):
        """Convert peak locations to steps per second"""
        if len(peak_info) == 0:
            num_seconds = int(np.round(acc.shape[0] / self.fs))
            return pd.Series(np.zeros(num_seconds))
        
        peak_info_count = np.ceil(peak_info / self.fs).astype(int).tolist()
        peak_info_count_dict = {x: peak_info_count.count(x) for x in set(peak_info_count)}
        
        num_seconds = int(np.floor(acc.shape[0] / self.fs))
        all_steps = pd.Series(range(num_seconds))
        return all_steps.map(peak_info_count_dict).fillna(0)
    
    def process_acceleration(self, raw_acc):
        """
        Main processing function
        
        Parameters:
        -----------
        raw_acc : numpy array
            Raw 3-axis accelerometer data (N x 3) at sampling frequency fs
        
        Returns:
        --------
        steps_per_sec : pandas Series
            Number of steps detected in each second
        """
        # Calculate vector magnitude
        acc = np.sqrt(np.sum(raw_acc ** 2, axis=1))
        
        # Check if acceleration is too low (no movement)
        if np.std(acc) < 0.025:
            num_seconds = int(np.round(len(acc) / self.fs))
            return pd.Series(np.zeros(num_seconds))
        
        # Apply step detection pipeline
        peak_info = self.find_peak(acc)
        
        if len(peak_info) == 0:
            num_seconds = int(np.round(len(acc) / self.fs))
            return pd.Series(np.zeros(num_seconds))
        
        peak_info = self.filter_magnitude(peak_info)
        
        if len(peak_info) < 2:
            num_seconds = int(np.round(len(acc) / self.fs))
            return pd.Series(np.zeros(num_seconds))
        
        peak_info = self.calc_periodicity(peak_info)
        
        if len(peak_info) < 3:
            num_seconds = int(np.round(len(acc) / self.fs))
            return pd.Series(np.zeros(num_seconds))
        
        peak_info = self.calc_similarity(peak_info)
        peak_info = self.filter_continuity(peak_info, acc)
        steps_per_sec = self.count_steps_per_second(peak_info, acc)
        
        return steps_per_sec


class ActiGraphDataProcessor:
    """
    Handles reading and resampling ActiGraph data files
    """
    
    def __init__(self, target_fs=15, source_fs=80):
        """
        Parameters:
        -----------
        target_fs : int
            Target sampling frequency (default: 15 Hz)
        source_fs : int
            Source sampling frequency (default: 80 Hz)
        """
        self.target_fs = target_fs
        self.source_fs = source_fs
        self.resample_interval = f"{1000/target_fs:.2f}ms"
    
    def read_actigraph_csv(self, filename):
        """
        Read ActiGraph CSV file with metadata header
        
        Returns:
        --------
        df : pandas DataFrame
            DataFrame with dateTime and accelerometer columns
        start_datetime : datetime
            Start datetime object
        """
        with open(filename) as f:
            lines = f.readlines()
        
        start_date = ""
        start_time = ""
        
        # Parse metadata
        for i, line in enumerate(lines):
            if line.startswith("Start Time"):
                start_time = line.split()[-1].strip().strip('"')
            elif line.startswith("Start Date"):
                start_date = line.split()[-1].strip().strip('"')
            elif line.startswith("Accelerometer X"):
                data_start_line = i
                break
        
        # Parse start datetime
        start_datetime = datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M:%S")
        
        # Read accelerometer data
        data = pd.read_csv(filename, skiprows=data_start_line, header=0)
        
        # Create timestamps
        step = timedelta(seconds=1 / self.source_fs)
        timestamps = [start_datetime + i * step for i in range(len(data))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'dateTime': timestamps,
            'acc_x': data.iloc[:, 0],
            'acc_y': data.iloc[:, 1],
            'acc_z': data.iloc[:, 2]
        })
        
        return df, start_datetime
    
    def resample_to_target_frequency(self, df):
        """
        Resample data from source frequency to target frequency
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame with dateTime index and accelerometer columns
        
        Returns:
        --------
        resampled_data : numpy array
            Resampled accelerometer data (N x 3)
        timestamps : pandas DatetimeIndex
            Corresponding timestamps
        """
        df_copy = df.copy()
        df_copy.set_index('dateTime', inplace=True)
        df_resampled = df_copy.resample(self.resample_interval).mean()
        df_resampled.reset_index(inplace=True)
        
        return df_resampled[['acc_x', 'acc_y', 'acc_z']].values, df_resampled['dateTime']
    
    def process_file(self, filepath):
        """
        Complete processing pipeline for a single file
        
        Returns:
        --------
        raw_acc : numpy array
            Resampled acceleration data
        timestamps : pandas DatetimeIndex
            Corresponding timestamps at target frequency
        start_datetime : datetime
            Start datetime of the file
        """
        df, start_datetime = self.read_actigraph_csv(filepath)
        raw_acc, timestamps = self.resample_to_target_frequency(df)
        return raw_acc, timestamps, start_datetime


def create_interval_output(steps_per_sec, start_datetime, interval_seconds=10):
    """
    Create output in START_TIME, STOP_TIME, STEP format with specified intervals
    
    Parameters:
    -----------
    steps_per_sec : pandas Series
        Steps counted per second
    start_datetime : datetime
        Start time of the recording
    interval_seconds : int
        Number of seconds per interval (default: 10)
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with columns: START_TIME, STOP_TIME, STEP
    """
    # Convert steps_per_sec to numpy array for easier manipulation
    steps_array = steps_per_sec.values
    total_seconds = len(steps_array)
    
    # Calculate number of complete intervals
    num_intervals = int(np.ceil(total_seconds / interval_seconds))
    
    results = []
    
    for i in range(num_intervals):
        start_idx = i * interval_seconds
        end_idx = min((i + 1) * interval_seconds, total_seconds)
        
        # Calculate time range
        start_time = start_datetime + timedelta(seconds=start_idx)
        end_time = start_datetime + timedelta(seconds=end_idx)
        
        # Sum steps in this interval
        step_count = int(np.sum(steps_array[start_idx:end_idx]))
        
        results.append({
            'START_TIME': start_time.strftime('%Y/%m/%d %H:%M:%S'),
            'STOP_TIME': end_time.strftime('%Y/%m/%d %H:%M:%S'),
            'STEP': step_count
        })
    
    return pd.DataFrame(results)


def process_single_file_with_output(file_path, output_dir, interval_seconds=10, verbose=True):
    """
    Process a single file and save output in the specified format
    
    Parameters:
    -----------
    file_path : str
        Path to the input file
    output_dir : str
        Directory to save output
    interval_seconds : int
        Number of seconds per interval (default: 10)
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    dict : Summary statistics for this file
    """
    if verbose:
        print(f"\nProcessing: {file_path}")
    
    # Initialize processors
    processor = ActiGraphDataProcessor(target_fs=15, source_fs=80)
    counter = VerisenseStepCounter()
    
    try:
        # Process file
        raw_acc, timestamps, start_datetime = processor.process_file(file_path)
        
        if len(raw_acc) == 0:
            print(f"  Warning: No data in {file_path}")
            return None
        
        # Count steps
        steps_per_sec = counter.process_acceleration(raw_acc)
        total_steps = int(steps_per_sec.sum())
        
        if verbose:
            print(f"  Total steps: {total_steps}")
            print(f"  Duration: {len(steps_per_sec)} seconds")
        
        # Create interval output
        output_df = create_interval_output(steps_per_sec, start_datetime, interval_seconds)
        
        # Generate output filename
        input_filename = Path(file_path).stem
        output_filename = f"{input_filename}_steps.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        
        if verbose:
            print(f"  Saved to: {output_path}")
        
        return {
            'file': file_path,
            'output': output_path,
            'total_steps': total_steps,
            'duration_seconds': len(steps_per_sec),
            'start_time': start_datetime.strftime('%Y/%m/%d %H:%M:%S'),
            'end_time': (start_datetime + timedelta(seconds=len(steps_per_sec))).strftime('%Y/%m/%d %H:%M:%S'),
            'num_intervals': len(output_df)
        }
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_multiple_files(file_paths, output_dir, interval_seconds=10, verbose=True):
    """
    Process multiple ActiGraph files and save individual outputs
    
    Parameters:
    -----------
    file_paths : list of str
        List of file paths to process
    output_dir : str
        Directory to save results
    interval_seconds : int
        Number of seconds per interval (default: 10)
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    results : dict
        Dictionary containing summary of all processed files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    file_results = []
    total_steps_all = 0
    
    for file_path in file_paths:
        result = process_single_file_with_output(
            file_path, 
            output_dir, 
            interval_seconds=interval_seconds,
            verbose=verbose
        )
        
        if result:
            file_results.append(result)
            total_steps_all += result['total_steps']
    
    # Create summary file
    if file_results:
        summary_df = pd.DataFrame(file_results)
        summary_path = os.path.join(output_dir, '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        if verbose:
            print(f"\nSummary saved to: {summary_path}")
    
    return {
        'total_steps_all_files': total_steps_all,
        'files_processed': len(file_results),
        'file_results': file_results
    }


def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(
        description='Process ActiGraph accelerometer data and count steps using Verisense algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file with 10-second intervals
  python stepcount_unified.py input.csv -o output_dir
  
  # Process multiple files from a list
  python stepcount_unified.py file_list.txt -o output_dir
  
  # Use 30-second intervals
  python stepcount_unified.py file_list.txt -o output_dir -i 30
        """
    )
    parser.add_argument(
        'input',
        help='Text file containing list of CSV file paths (one per line) or single CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory for results (required)',
        required=True
    )
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=10,
        help='Interval length in seconds for output (default: 10)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output'
    )
    
    args = parser.parse_args()
    
    # Determine input type
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    # Read file paths
    if input_path.suffix == '.csv':
        # Single CSV file
        file_paths = [str(input_path)]
    else:
        # Text file with list of paths
        with open(input_path, 'r') as f:
            file_paths = [line.strip().strip('"') for line in f if line.strip()]
    
    if not file_paths:
        print("Error: No valid file paths found")
        sys.exit(1)
    
    print(f"Processing {len(file_paths)} file(s)...")
    print(f"Output directory: {args.output}")
    print(f"Interval length: {args.interval} seconds")
    
    # Process files
    results = process_multiple_files(
        file_paths, 
        output_dir=args.output, 
        interval_seconds=args.interval,
        verbose=args.verbose
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Files processed: {results['files_processed']}")
    print(f"Total steps (all files): {results['total_steps_all_files']:,}")
    print(f"\nIndividual file outputs saved in: {args.output}")
    print(f"Summary file: {os.path.join(args.output, '_summary.csv')}")
    print("="*60)


if __name__ == "__main__":
    main()