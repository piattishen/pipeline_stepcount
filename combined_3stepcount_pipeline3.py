"""
Combined Step Counting Pipeline — Linux / SBATCH compatible
===========================================================
Algorithms
  1. Oxford    – calls stepcount CLI directly (must be active conda env)
  2. OAK       – pure Python, CWT-based
  3. Verisense – pure Python, peak-detection-based

Usage
-----
  python combined_3stepcount_pipeline.py <input_dir> <output_dir>

  Example (single subject):
    python combined_3stepcount_pipeline.py \\
        /scratch/wang.yichen8/PAAWS_FreeLiving/DS_10/accel \\
        /scratch/wang.yichen8/PAAWS_results

  SBATCH array usage:
    subjects=(DS_10 DS_11 DS_12 ...)
    subj=${subjects[$SLURM_ARRAY_TASK_ID - 1]}
    python combined_3stepcount_pipeline.py \\
        /scratch/wang.yichen8/PAAWS_FreeLiving/$subj/accel \\
        /scratch/wang.yichen8/PAAWS_results

Output structure
----------------
  <output_dir>/
    <subject>/
      oxford/
        oxford_intermediates/         <- stepcount CLI working files
          <filename>.csv
          outputs/
            <filename>/
              <filename>-Steps.csv.gz <- raw stepcount output
        <filename>_steps.csv          <- reformatted: time,Steps
      oak/
        <filename>_steps.csv          <- reformatted: time,Steps
      verisense/
        <filename>_steps.csv          <- reformatted: time,Steps
      _ALL_FILES_SUMMARY.csv

Output format (ALL algorithms, unified)
----------------------------------------
  time,Steps
  2021-11-01 18:00:00,0.0
  2021-11-01 18:00:10,5.0
  ...
"""

import argparse
import os
import sys
import subprocess
import glob
import zipfile
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

warnings.filterwarnings("ignore")


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _find_header_line(lines: list) -> int:
    """
    Robustly find the row index of 'Accelerometer X,...' header.
    Handles BOM, CRLF, and extra whitespace.
    """
    for i, line in enumerate(lines):
        cleaned = line.strip().lstrip("\ufeff")
        if cleaned.startswith("Accelerometer X"):
            return i
    return None


def read_actigraph(filepath: str, source_hz: int = 80) -> tuple:
    """
    Parse a raw ActiGraph GT3X+ CSV file.

    Expected format:
        ------------ Data File Created By ActiGraph ...
        Start Time 18:00:00
        Start Date 11/1/2021
        ...
        Accelerometer X,Accelerometer Y,Accelerometer Z
        0.000,0.000,0.000
        ...

    Returns
    -------
    data      : pd.DataFrame  columns x, y, z
    start_dt  : datetime
    actual_hz : int
    """
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    start_date = start_time = ""
    actual_hz  = source_hz

    for line in lines:
        s = line.strip()
        if s.startswith("Start Time"):
            start_time = s.split()[-1].strip().strip('"')
        elif s.startswith("Start Date"):
            start_date = s.split()[-1].strip().strip('"')
        elif "Hz" in s and "at" in s:
            try:
                parts  = s.split()
                hz_idx = next(j for j, p in enumerate(parts) if p == "Hz")
                actual_hz = int(parts[hz_idx - 1])
            except Exception:
                pass

    data_line = _find_header_line(lines)
    if data_line is None:
        raise ValueError(f"Cannot find 'Accelerometer X' header in {filepath}")

    # Parse M/d/yyyy date (no zero-padding) robustly
    start_dt = _parse_actigraph_datetime(start_date, start_time, filepath)

    data = pd.read_csv(
        filepath, skiprows=data_line, header=0,
        usecols=[0, 1, 2], names=["x", "y", "z"],
        encoding="utf-8-sig", on_bad_lines="skip"
    )
    data = data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    return data, start_dt, actual_hz


def _parse_actigraph_datetime(start_date: str, start_time: str,
                               filepath: str = "") -> datetime:
    """Parse ActiGraph M/d/yyyy date + HH:MM:SS time robustly."""
    dt_str = f"{start_date} {start_time}".strip()
    # Try standard formats first
    for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            pass
    # Fallback: manual split to handle M/d/yyyy without zero padding
    try:
        date_part, time_part = dt_str.split(" ", 1)
        m, d, y = date_part.split("/")
        return datetime.strptime(
            f"{int(m):02d}/{int(d):02d}/{y} {time_part}", "%m/%d/%Y %H:%M:%S"
        )
    except Exception as e:
        raise ValueError(
            f"Cannot parse datetime '{dt_str}' in {filepath}: {e}"
        )


def build_output_df(steps_per_sec: np.ndarray, start_dt: datetime,
                    interval: int = 10) -> pd.DataFrame:
    """
    Aggregate per-second step counts into fixed-width intervals.

    Output columns: time, Steps
    Output format:  2021-11-01 18:00:00, 5.0
    Matches Oxford stepcount native output format exactly.
    """
    n    = len(steps_per_sec)
    rows = []
    for i in range(int(np.ceil(n / interval))):
        s  = i * interval
        e  = min(s + interval, n)
        t0 = start_dt + timedelta(seconds=s)
        rows.append({
            "time":  t0.strftime("%Y-%m-%d %H:%M:%S"),
            "Steps": float(int(np.sum(steps_per_sec[s:e])))
        })
    return pd.DataFrame(rows)


def save_output(df: pd.DataFrame, out_dir: str, stem: str,
                suffix: str = "steps") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}_{suffix}.csv")
    df.to_csv(path, index=False)
    return path


# ============================================================================
# ALGORITHM 1 — Oxford  (external stepcount CLI)
# ============================================================================

def _prepare_oxford_csv(filepath: str, intermediates_dir: str) -> str:
    """
    Read raw ActiGraph CSV, add 'time' column, save to intermediates_dir.
    This is what the stepcount CLI consumes.
    """
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    start_date = start_time = ""
    for line in lines:
        s = line.strip()
        if s.startswith("Start Time"):
            start_time = s.split()[-1].strip().strip('"')
        elif s.startswith("Start Date"):
            start_date = s.split()[-1].strip().strip('"')

    data_line = _find_header_line(lines)
    if data_line is None:
        raise ValueError(f"Cannot find 'Accelerometer X' header in {filepath}")

    start_dt = _parse_actigraph_datetime(start_date, start_time, filepath)

    data = pd.read_csv(
        filepath, skiprows=data_line, header=0,
        encoding="utf-8-sig", on_bad_lines="skip"
    )
    data = data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    step = timedelta(seconds=1 / 80)
    timestamps = [start_dt + i * step for i in range(len(data))]

    df = data.copy()
    df.insert(0, "time", timestamps)

    os.makedirs(intermediates_dir, exist_ok=True)
    out_path = os.path.join(
        intermediates_dir,
        os.path.splitext(os.path.basename(filepath))[0] + ".csv"
    )
    df.to_csv(out_path, index=False)
    print(f"    Oxford intermediate saved to {out_path}")
    return out_path


def run_oxford(filepath: str, out_dir: str, stem: str,
               interval: int, source_hz: int,
               oxford_output_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # Step A: prepare intermediate CSV
    intermediates_dir = os.path.join(oxford_output_dir, "oxford_intermediates")
    try:
        intermediate_csv = _prepare_oxford_csv(filepath, intermediates_dir)
    except Exception as e:
        return {"status": "error", "msg": f"Oxford CSV prep failed: {e}"}

    # Step B: run stepcount CLI
    cmd = [
        "stepcount", intermediate_csv,
        "--txyz", "time,Accelerometer X,Accelerometer Y,Accelerometer Z"
    ]
    print(f"    Running: {' '.join(cmd)}")
    # Run from intermediates_dir so stepcount writes outputs/ subfolder there
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=intermediates_dir)

    stdout_tail = result.stdout[-500:] if result.stdout else ""

    # Step C: find *-Steps.csv.gz or *-Steps.csv anywhere under intermediates_dir
    # stepcount writes to: intermediates_dir/outputs/<stem>/<stem>-Steps.csv.gz
    steps_csv_path = None

    # 1. .csv.gz (stepcount default)
    matches = glob.glob(
        os.path.join(intermediates_dir, "**", "*-Steps.csv.gz"), recursive=True
    )
    if matches:
        exact = [m for m in matches if stem in os.path.basename(m)]
        steps_csv_path = exact[0] if exact else matches[0]

    # 2. Plain .csv
    if not steps_csv_path:
        matches = glob.glob(
            os.path.join(intermediates_dir, "**", "*-Steps.csv"), recursive=True
        )
        if matches:
            exact = [m for m in matches if stem in os.path.basename(m)]
            steps_csv_path = exact[0] if exact else matches[0]

    # 3. Inside a .zip archive
    if not steps_csv_path:
        for zp in glob.glob(
            os.path.join(intermediates_dir, "**", "*.zip"), recursive=True
        ):
            with zipfile.ZipFile(zp, "r") as zf:
                inner = [n for n in zf.namelist() if n.endswith("-Steps.csv")]
                if inner:
                    extract_to = os.path.join(intermediates_dir, "_extracted")
                    zf.extract(inner[0], extract_to)
                    steps_csv_path = os.path.join(extract_to, inner[0])
                    break

    if not steps_csv_path or not os.path.isfile(steps_csv_path):
        return {
            "status": "error",
            "msg": (f"Could not find *-Steps.csv(.gz) in {intermediates_dir}.\n"
                    f"stdout: {stdout_tail}")
        }

    # Step D: read .gz (pandas handles automatically) and reformat to time,Steps
    try:
        sc_df = pd.read_csv(steps_csv_path, parse_dates=["time"])
        sc_df = sc_df.sort_values("time").reset_index(drop=True)
        sc_df["time"]  = sc_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        sc_df["Steps"] = sc_df["Steps"].apply(lambda x: float(int(round(x))))
        out_df = sc_df[["time", "Steps"]].copy()

    except Exception as e:
        return {"status": "error", "msg": f"Output parse error: {e}"}

    path  = save_output(out_df, out_dir, stem)
    total = int(out_df["Steps"].sum())
    return {"status": "success", "total_steps": total, "path": path}


# ============================================================================
# ALGORITHM 2 — OAK  (CWT-based, pure Python)
# ============================================================================

def _adjust_bout(arr: np.ndarray, fs: int = 10) -> np.ndarray:
    if len(arr) % fs >= 0.7 * fs:
        arr = np.append(arr, np.full(fs - len(arr) % fs, arr[-1]))
    else:
        arr = arr[:len(arr) // fs * fs]
    return arr


def _preprocess_bout(t, x, y, z, fs=10):
    if any(len(a) < 2 for a in [t, x, y, z]):
        return np.array([]), np.array([])

    t0 = t - t[0]
    ti = np.arange(t0[0], t0[-1], 1 / fs) + t[0]

    xi = interpolate.interp1d(t, x)(ti)
    yi = interpolate.interp1d(t, y)(ti)
    zi = interpolate.interp1d(t, z)(ti)

    xi = _adjust_bout(xi, fs)
    yi = _adjust_bout(yi, fs)
    zi = _adjust_bout(zi, fs)

    n_sec = int(np.floor(len(xi) / fs))
    ti    = ti[:n_sec * fs][::fs]

    vm = np.sqrt(xi**2 + yi**2 + zi**2)
    if vm.shape[0] > 0 and np.mean(vm) > 5:
        xi /= 9.80665; yi /= 9.80665; zi /= 9.80665

    vm = np.sqrt(xi**2 + yi**2 + zi**2) - 1
    return ti, vm


def _get_pp(vm, fs=10):
    return np.ptp(vm.reshape((fs, -1), order="F"), axis=0)


def _cwt_and_peaks(vm, fs=10, step_freq=(1.4, 2.3), alpha=0.6, beta=2.5):
    try:
        from ssqueezepy import ssq_cwt
    except ImportError:
        raise ImportError(
            "ssqueezepy is required for OAK. Install: pip install ssqueezepy"
        )

    wavelet = ("gmw", {"beta": 90, "gamma": 3})
    window  = tukey(len(vm), alpha=0.02, sym=True)
    padded  = np.concatenate([np.zeros(5 * fs), vm * window, np.zeros(5 * fs)])

    out   = ssq_cwt(padded[:-1], wavelet, fs=fs)
    coefs = np.abs(out[0] ** 2).astype("complex128").real
    coefs = np.append(coefs, coefs[:, -1:], axis=1)
    freqs = out[2]

    fi  = np.arange(0.5, 4.5, 0.05)
    itp = interpolate.RegularGridInterpolator(
              (freqs, np.arange(coefs.shape[1])), coefs)
    gx, gy = np.meshgrid(fi, np.arange(coefs.shape[1]), indexing="ij")
    ci  = itp((gx, gy))[:, 5 * fs: -5 * fs]

    n_rows, n_cols = ci.shape
    n_sec = n_cols // fs
    dp    = np.zeros((n_rows, n_sec))
    lo    = np.argmin(abs(fi - step_freq[0]))
    hi    = np.argmin(abs(fi - step_freq[1]))

    for i in range(n_sec):
        win      = np.sum(ci[:, i * fs:(i + 1) * fs], axis=1)
        locs, _  = find_peaks(win)
        if len(locs) == 0:
            continue
        pks   = win[locs]
        order = np.argsort(-pks)
        locs, pks = locs[order], pks[order]

        idx_in_range = next((j for j, l in enumerate(locs)
                             if lo <= l <= hi), None)
        if idx_in_range is None:
            continue

        pk_in  = pks[idx_in_range]
        pks_lo = pks[locs < lo]
        pks_hi = pks[locs > hi]
        max_lo = np.max(pks_lo) if len(pks_lo) else 0
        max_hi = np.max(pks_hi) if len(pks_hi) else 0

        if max_hi / pk_in < beta or max_lo / pk_in < alpha:
            dp[locs[idx_in_range], i] = 1
    return fi, ci, dp


def _find_cont_peaks(valid_peaks, min_t=3, delta=20):
    n_rows, n_cols = valid_peaks.shape
    ext = np.zeros((n_rows, n_cols + 1), dtype=valid_peaks.dtype)
    ext[:, :n_cols] = valid_peaks
    cont = np.zeros_like(ext)

    for si in range(n_cols + 1 - min_t):
        sl   = ext[:, si:si + min_t].copy()
        wins = list(range(min_t)) + list(range(min_t - 2, -1, -1))
        stop = True

        for wi in wins:
            pr   = np.where(sl[:, wi] != 0)[0]
            stop = True
            for p in pr:
                idx = np.arange(max(0, p - delta), min(p + delta + 1, n_rows))
                p1  = p2 = sl[p, wi]
                if wi == 0:
                    p1 = p1 + sl[idx, wi + 1]
                elif wi == min_t - 1:
                    p1 = p1 + sl[idx, wi - 1]
                else:
                    p1 = p1 + sl[idx, wi - 1]
                    p2 = p2 + sl[idx, wi + 1]

                if wi in (0, min_t - 1):
                    if np.any(p1 > 1): stop = False
                    else:              sl[p, wi] = 0
                else:
                    if np.any(p1 > 1) and np.any(p2 > 1): stop = False
                    else: sl[p, wi] = 0
            if stop:
                break

        if not stop:
            cont[:, si:si + min_t] = sl

    return cont[:, :-1]


def _find_walking(vm, fs=10, min_amp=0.3, step_freq=(1.4, 2.3),
                  alpha=0.6, beta=2.5, min_t=3, delta=20):
    pp    = _get_pp(vm, fs)
    valid = pp >= min_amp

    if valid.sum() < min_t:
        return np.zeros(len(pp))

    fi, ci, dp = _cwt_and_peaks(vm[np.repeat(valid, fs)], fs,
                                 step_freq, alpha, beta)

    vp = np.zeros((dp.shape[0], len(valid)))
    vp[:, valid] = dp
    cp  = _find_cont_peaks(vp, min_t, delta)

    cad = np.zeros(len(valid))
    for i in range(len(cad)):
        idx = np.where(cp[:, i] > 0)[0]
        if len(idx):
            cad[i] = fi[idx[0]]
    return cad


def run_oak(filepath: str, out_dir: str, stem: str,
            interval: int, source_hz: int,
            downsample_method: str = "decimate") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    try:
        raw, start_dt, hz = read_actigraph(filepath, source_hz)
    except Exception as e:
        return {"status": "error", "msg": str(e)}

    factor = hz // 10
    if downsample_method == "average":
        n_full = len(raw) // factor * factor
        x10 = raw["x"].values[:n_full].reshape(-1, factor).mean(axis=1)
        y10 = raw["y"].values[:n_full].reshape(-1, factor).mean(axis=1)
        z10 = raw["z"].values[:n_full].reshape(-1, factor).mean(axis=1)
    else:
        x10 = raw["x"].values[::factor]
        y10 = raw["y"].values[::factor]
        z10 = raw["z"].values[::factor]

    n10 = len(x10)
    t10 = np.array([start_dt.timestamp() + i * 0.1 for i in range(n10)])

    try:
        t_i, vm = _preprocess_bout(t10, x10, y10, z10, fs=10)
    except Exception as e:
        return {"status": "error", "msg": f"Preprocess error: {e}"}

    if len(t_i) == 0:
        return {"status": "error", "msg": "Empty data after preprocessing"}

    try:
        cad = _find_walking(vm, fs=10, min_amp=0.3,
                            step_freq=(1.4, 2.3), alpha=0.6,
                            beta=2.5, min_t=3, delta=20)
    except Exception as e:
        return {"status": "error", "msg": f"Walking detection error: {e}"}

    steps_per_sec = np.round(cad).astype(int).astype(float)
    out_df = build_output_df(steps_per_sec, start_dt, interval)
    path   = save_output(out_df, out_dir, stem)

    total = int(out_df["Steps"].sum())
    return {"status": "success", "total_steps": total, "path": path}


# ============================================================================
# ALGORITHM 3 — Verisense  (peak-detection, pure Python)
# ============================================================================

class _VerisenseCounter:
    def __init__(self, peak_win_len=3, period_min=5, period_max=15,
                 sim_thres=-0.5, cont_win_size=3, cont_thres=4,
                 var_thres=0.001, mag_thres=1.2, fs=15):
        self.peak_win_len  = peak_win_len
        self.period_min    = period_min
        self.period_max    = period_max
        self.sim_thres     = sim_thres
        self.cont_win_size = cont_win_size
        self.cont_thres    = cont_thres
        self.var_thres     = var_thres
        self.mag_thres     = mag_thres
        self.fs            = fs

    def _find_peak(self, acc):
        k, hk = self.peak_win_len, round(self.peak_win_len / 2)
        n     = len(acc)
        segs  = int(np.floor(n / k))
        locs, mags = [], []
        for i in range(segs):
            s  = i * k; e = s + k
            lm = int(np.argmax(acc[s:e])); b = s + lm
            sc = max(0, b - hk); ec = min(n, b + hk + 1)
            if int(np.argmax(acc[sc:ec])) == b - sc:
                locs.append(b); mags.append(float(np.max(acc[s:e])))
        if not locs:
            return np.empty((0, 5))
        pi = np.full((len(locs), 5), np.nan)
        pi[:, 0] = locs; pi[:, 1] = mags
        return pi

    def _filter_mag(self, pi):
        return pi[pi[:, 1] > self.mag_thres]

    def _periodicity(self, pi):
        if len(pi) < 2: return np.empty((0, 5))
        pi[:-1, 2] = np.diff(pi[:, 0])
        return pi[(pi[:, 2] > self.period_min) & (pi[:, 2] < self.period_max)]

    def _similarity(self, pi):
        if len(pi) < 3: return np.empty((0, 5))
        pi[:-2, 3] = -np.abs(pi[2:, 1] - pi[:-2, 1])
        return pi[~np.isnan(pi[:, 3]) & (pi[:, 3] > self.sim_thres)]

    def _continuity(self, pi, acc):
        if len(pi) < self.cont_thres + 1: return np.array([])
        n    = len(pi); cont = np.zeros(n)
        for i in range(self.cont_thres - 1, n - 1):
            v = 0
            for x in range(1, self.cont_thres + 1):
                seg = acc[int(pi[i-x, 0]):min(int(pi[i-x+1, 0])+1, len(acc))]
                if len(seg) > 1 and np.var(seg, ddof=1) > self.var_thres:
                    v += 1
            cont[i] = int(v >= self.cont_win_size)
        locs = pi[cont == 1, 0]
        return locs[~np.isnan(locs)]

    def _count_per_sec(self, locs, acc):
        n_sec = int(np.round(len(acc) / self.fs))
        if len(locs) == 0: return np.zeros(n_sec)
        idx    = np.clip((np.array(locs) // self.fs).astype(int), 0, n_sec - 1)
        counts = np.zeros(n_sec)
        for i in idx: counts[i] += 1
        return counts

    def process(self, raw_acc: np.ndarray) -> np.ndarray:
        acc   = np.sqrt(np.sum(raw_acc**2, axis=1))
        n_sec = int(np.round(len(acc) / self.fs))
        zeros = np.zeros(n_sec)

        if np.std(acc, ddof=1) < 0.025: return zeros
        pi = self._find_peak(acc)
        if len(pi) < 2:  return zeros
        pi = self._filter_mag(pi)
        if len(pi) < 3:  return zeros
        pi = self._periodicity(pi)
        if len(pi) < 3:  return zeros
        pi = self._similarity(pi)
        if len(pi) == 0: return zeros
        locs = self._continuity(pi, acc)
        return self._count_per_sec(locs, acc)


def run_verisense(filepath: str, out_dir: str, stem: str,
                  interval: int, source_hz: int) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    try:
        raw, start_dt, hz = read_actigraph(filepath, source_hz)
    except Exception as e:
        return {"status": "error", "msg": str(e)}

    step_td   = timedelta(seconds=1 / hz)
    ts        = [start_dt + i * step_td for i in range(len(raw))]
    raw.index = pd.DatetimeIndex(ts)
    us        = int(round(1_000_000 / 15))
    resampled = raw.resample(f"{us}us").mean().interpolate(method="time")
    raw_acc   = resampled[["x", "y", "z"]].values

    counter       = _VerisenseCounter(fs=15)
    steps_per_sec = counter.process(raw_acc)

    out_df = build_output_df(steps_per_sec, start_dt, interval)
    path   = save_output(out_df, out_dir, stem)
    total  = int(out_df["Steps"].sum())
    return {"status": "success", "total_steps": total, "path": path}


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

def process_file(filepath: str, subject_out: str, algorithms: list,
                 interval: int, source_hz: int,
                 oxford_output_dir: str,
                 downsample_method: str) -> dict:
    stem   = os.path.splitext(os.path.basename(filepath))[0]
    record = {"file": os.path.basename(filepath)}

    for algo in algorithms:
        out_dir = os.path.join(subject_out, algo)
        print(f"  [{algo}] {stem} ...", end=" ", flush=True)

        if algo == "oxford":
            res = run_oxford(filepath, out_dir, stem, interval,
                             source_hz, oxford_output_dir)
        elif algo == "oak":
            res = run_oak(filepath, out_dir, stem, interval,
                          source_hz, downsample_method)
        elif algo == "verisense":
            res = run_verisense(filepath, out_dir, stem, interval, source_hz)
        else:
            res = {"status": "error", "msg": f"Unknown algorithm: {algo}"}

        if res["status"] == "success":
            print(f"✓  {res['total_steps']:,} steps")
        else:
            print(f"✗  {res['msg']}")

        record[f"{algo}_steps"]  = res.get("total_steps", "error")
        record[f"{algo}_status"] = res["status"]

    return record


def run_pipeline(input_dir: str, output_dir: str, subject: str,
                 algorithms: list, interval: int, source_hz: int,
                 oxford_output_dir: str, downsample_method: str):

    csvs = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".csv"))
    if not csvs:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    subject_out = os.path.join(output_dir, subject)
    os.makedirs(subject_out, exist_ok=True)

    print(f"\nSubject  : {subject}")
    print(f"Files    : {len(csvs)}")
    print(f"Algos    : {', '.join(algorithms)}")
    print(f"Output   : {subject_out}")
    if "oxford" in algorithms:
        print(f"Oxford rawout : {oxford_output_dir}")
    print()

    summary_rows = []
    for fp in csvs:
        full = os.path.join(input_dir, fp)
        print(f"▶ {fp}")
        row = process_file(full, subject_out, algorithms, interval,
                           source_hz, oxford_output_dir, downsample_method)
        summary_rows.append(row)

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(subject_out, "_ALL_FILES_SUMMARY.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Combined step counting pipeline (Oxford / OAK / Verisense)",
        usage="python combined_3stepcount_pipeline.py <input_dir> <output_dir>"
    )
    p.add_argument("input_dir",
                   help="Directory containing raw ActiGraph CSV files "
                        "(e.g. /scratch/.../DS_10/accel)")
    p.add_argument("output_dir",
                   help="Root output directory (e.g. /scratch/.../PAAWS_results)")
    return p.parse_args()


# ── Constants — edit these to match your setup ───────────────────────────────
ALGORITHMS        = ["oxford", "oak", "verisense"]
INTERVAL          = 10          # output interval in seconds
SOURCE_HZ         = 80          # ActiGraph sample rate
DOWNSAMPLE_METHOD = "decimate"  # "decimate" or "average" for OAK 80→10 Hz


if __name__ == "__main__":
    args    = parse_args()
    # subject name = parent folder of input_dir
    # e.g. /scratch/.../PAAWS_FreeLiving/DS_10/accel  →  DS_10
    subject = os.path.basename(os.path.dirname(os.path.normpath(args.input_dir)))

    oxford_output_dir = os.path.join(args.output_dir, subject, "oxford")
    os.makedirs(oxford_output_dir, exist_ok=True)

    run_pipeline(
        input_dir         = args.input_dir,
        output_dir        = args.output_dir,
        subject           = subject,
        algorithms        = ALGORITHMS,
        interval          = INTERVAL,
        source_hz         = SOURCE_HZ,
        oxford_output_dir = oxford_output_dir,
        downsample_method = DOWNSAMPLE_METHOD,
    )