#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         STEP ALGORITHM COMPARISON ANALYSIS  —  Cluster Version      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Auto-discovers all DS_xx folders, all sensors, all 3 algorithms    ║
║  (oxford / oak / verisense), reads .csv.gz files, joins with        ║
║  activity labels, and outputs summary CSVs + comparison plots.      ║
╠══════════════════════════════════════════════════════════════════════╣
║  USAGE                                                               ║
║    python step_analysis.py                                           ║
║    python step_analysis.py --root /scratch/wang.yichen8/PAAWS_results║
║    python step_analysis.py --ds DS_10 DS_12    # specific DS only   ║
║    python step_analysis.py --out ./my_results  # custom output dir  ║
╠══════════════════════════════════════════════════════════════════════╣
║  EXPECTED DIRECTORY LAYOUT                                           ║
║    {root}/                                                           ║
║      DS_10/                                                          ║
║        oxford/oxford_intermediates/outputs/                          ║
║          DS_10-Free-LeftWrist/                                       ║
║            DS_10-Free-LeftWrist-Steps.csv.gz   <- sensor file       ║
║          DS_10-Free-RightWrist/                                      ║
║        oak/                                                          ║
║          DS_10-Free-LeftWrist-Steps.csv(.gz)   <- flat or nested    ║
║        verisense/                                                    ║
║          DS_10-Free-LeftWrist-Steps.csv(.gz)                         ║
║        labels/                                                       ║
║          DS_10-Free-label.csv   (or .csv.gz)                        ║
║      DS_12/                                                          ║
║        ...                                                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, re, glob, argparse, warnings, gzip, io
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ROOT   = "/scratch/wang.yichen8/PAAWS_results"
DEFAULT_OUTPUT = "/scratch/wang.yichen8/step_analysis_output"
DEFAULT_LABELS = "/scratch/wang.yichen8/PAAWS_FreeLiving"
ALGORITHMS     = ["oxford", "oak", "verisense"]

LABEL_PATTERNS = [
    "{ds_root}/labels/{ds_id}-Free-label.csv",
    "{ds_root}/labels/{ds_id}-Free-label.csv.gz",
    "{ds_root}/{ds_id}-Free-label.csv",
    "{ds_root}/{ds_id}-Free-label.csv.gz",
]

WALKING_ACTIVITIES = [
    "Walking", "Walking_Fast", "Walking_Slow",
    "Walking_Up_Stairs", "Walking_Down_Stairs",
]
NON_WALKING_ACTIVITIES = [
    "Sitting_With_Movement", "Standing_With_Movement", "Standing_Still",
    "Puttering_Around", "Lying_Still", "Lying_With_Movement",
    "Washing_Hands", "Brushing_Teeth", "Folding_Clothes", "Ironing",
    "Sweeping", "Vacuuming", "Kneeling_With_Movement",
    "Brushing/Combing/Tying_Hair", "Watering_Plants",
    "Organizing_Shelf/Cabinet", "Putting_Clothes_Away",
    "Loading/Unloading_Washing_Machine/Dryer",
]
EXCLUDE_ACTIVITIES = [
    "PA_Type_Video_Unavailable/Indecipherable",
    "Posture_Video_Unavailable/Indecipherable",
    "Synchronizing_Sensors", "PA_Type_Too_Complex", "PA_Type_Other",
]
MIN_SEGMENT_SEC = 10

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────

ALGO_COLORS  = {"oxford": "#2E86AB", "oak": "#E84855", "verisense": "#57CC99"}
EXTRA_COLORS = ["#F4A261", "#9B5DE5", "#F15BB5", "#FEE440"]
DARK_BG, DARK_AX, GRID_COL = "#0F1117", "#111827", "#2a2a3e"

def algo_color(name):
    for key, col in ALGO_COLORS.items():
        if key in name.lower():
            return col
    return EXTRA_COLORS[hash(name) % len(EXTRA_COLORS)]

# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_ds_folders(root):
    folders = sorted([d for d in glob.glob(os.path.join(root, "DS_*")) if os.path.isdir(d)])
    if not folders:
        raise FileNotFoundError(f"No DS_* folders found under: {root}")
    return folders


def find_label_file(ds_root, ds_id, label_root=None):
    """
    Search for the label CSV. Checks two places:
      1. label_root/{ds_id}/label/{ds_id}-Free-label.csv  (separate raw data folder)
      2. ds_root itself (fallback patterns)
    """
    # Primary: separate label root (e.g. /scratch/.../PAAWS_FreeLiving/DS_10/label/)
    if label_root:
        candidates = [
            os.path.join(label_root, ds_id, "label", f"{ds_id}-Free-label.csv"),
            os.path.join(label_root, ds_id, "label", f"{ds_id}-Free-label.csv.gz"),
            os.path.join(label_root, ds_id, f"{ds_id}-Free-label.csv"),
            os.path.join(label_root, ds_id, f"{ds_id}-Free-label.csv.gz"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path

    # Fallback: look inside the DS results folder itself
    for pat in LABEL_PATTERNS:
        path = pat.format(ds_root=ds_root, ds_id=ds_id)
        if os.path.isfile(path):
            return path
    for ext in ["csv", "csv.gz"]:
        hits = glob.glob(os.path.join(ds_root, "**", f"*label*.{ext}"), recursive=True)
        if hits:
            return hits[0]
    return None


def find_steps_files(ds_root, ds_id, algorithm):
    """
    Return list of (sensor_name, filepath) for the given algorithm.
    Handles Oxford's deep nesting AND flat oak/verisense layouts.
    """
    algo_dir = os.path.join(ds_root, algorithm)
    if not os.path.isdir(algo_dir):
        return []

    hits = []
    for ext in ["csv.gz", "csv"]:
        hits += glob.glob(os.path.join(algo_dir, "**", f"*[Ss]teps.{ext}"), recursive=True)
        hits += glob.glob(os.path.join(algo_dir, f"*[Ss]teps.{ext}"))

    # Prefer .gz over .csv when both exist for the same stem
    seen = {}
    for h in hits:
        stem = re.sub(r"\.csv(\.gz)?$", "", os.path.basename(h))
        if stem not in seen or h.endswith(".gz"):
            seen[stem] = h

    results = []
    for stem, path in seen.items():
        # Strip the steps suffix (-Steps, _steps, -steps, _Steps) then grab sensor name
        clean = re.sub(r"[-_][Ss]teps$", "", stem)
        m = re.search(r"(?:Free-)(.+)$", clean)
        sensor = m.group(1) if m else clean
        results.append((sensor, path))

    return sorted(results)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_auto(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pd.read_csv(io.BytesIO(f.read()))
    return pd.read_csv(path)


def load_labels(path):
    df = read_csv_auto(path)
    df["START_TIME"] = pd.to_datetime(df["START_TIME"])
    df["STOP_TIME"]  = pd.to_datetime(df["STOP_TIME"])
    df["duration_sec"] = (df["STOP_TIME"] - df["START_TIME"]).dt.total_seconds()
    df = df[df["duration_sec"] >= MIN_SEGMENT_SEC].copy()
    df = df[~df["PA_TYPE"].isin(EXCLUDE_ACTIVITIES)].copy()
    return df.dropna(subset=["PA_TYPE"]).reset_index(drop=True)


def load_steps_series(path):
    df = read_csv_auto(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c or "date" in c), df.columns[0])
    step_col = next((c for c in df.columns if "step" in c), df.columns[1])
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col)[step_col].sort_index()

# ─────────────────────────────────────────────────────────────────────────────
# CORE AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def activity_category(pa_type):
    if pa_type in WALKING_ACTIVITIES:     return "Walking"
    if pa_type in NON_WALKING_ACTIVITIES: return "Non-Walking"
    return "Other"


def aggregate_segments(labels_df, series, ds_id, algorithm, sensor):
    """
    Vectorized aggregation using numpy searchsorted on integer nanosecond timestamps.
    ~87x faster than row-by-row boolean masking.
    """
    # Convert datetime index and label boundaries to int64 nanoseconds once
    idx    = series.index.values.astype("int64")
    sv     = series.values.astype(float)
    starts = labels_df["START_TIME"].values.astype("int64")
    stops  = labels_df["STOP_TIME"].values.astype("int64")
    durs   = labels_df["duration_sec"].values

    records = []
    for i in range(len(labels_df)):
        lo    = np.searchsorted(idx, starts[i], side="left")
        hi    = np.searchsorted(idx, stops[i],  side="left")
        total = float(sv[lo:hi].sum())
        dur   = durs[i]
        spm   = (total / dur * 60) if dur > 0 else 0.0
        records.append({
            "ds_id": ds_id, "algorithm": algorithm, "sensor": sensor,
            "PA_TYPE":            labels_df.iloc[i]["PA_TYPE"],
            "activity_category":  activity_category(labels_df.iloc[i]["PA_TYPE"]),
            "duration_sec": dur, "total_steps": total, "steps_per_min": spm,
        })
    return records

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLES
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(df_long):
    grp = df_long.groupby(["ds_id", "sensor", "algorithm", "PA_TYPE", "activity_category"])
    return grp["steps_per_min"].agg(
        mean_spm="mean", std_spm="std", median_spm="median", n_segments="count"
    ).reset_index()


def build_ranking(df_long):
    rows = []
    for (ds_id, sensor, algo), sub in df_long.groupby(["ds_id", "sensor", "algorithm"]):
        w  = sub[sub["activity_category"] == "Walking"]["steps_per_min"]
        nw = sub[sub["activity_category"] == "Non-Walking"]
        rows.append({
            "ds_id": ds_id, "sensor": sensor, "algorithm": algo,
            "walk_mean_spm":    round(w.mean(), 2)  if len(w) else None,
            "walk_cv_pct":      round(w.std() / w.mean() * 100, 1)
                                if len(w) > 1 and w.mean() > 0 else None,
            "nonwalk_mean_spm": round(nw["steps_per_min"].mean(), 2) if len(nw) else None,
            "fp_rate_pct":      round((nw["total_steps"] > 0).mean() * 100, 1) if len(nw) else None,
            "n_walk_segments":   int((sub["activity_category"] == "Walking").sum()),
            "n_nonwalk_segments":int((sub["activity_category"] == "Non-Walking").sum()),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG, DARK_AX, GRID_COL = "#0F1117", "#111827", "#2a2a3e"

def algo_color(name):
    for key, col in ALGO_COLORS.items():
        if key in name.lower():
            return col
    return EXTRA_COLORS[hash(name) % len(EXTRA_COLORS)]

def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", axis="both")
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color="white", fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color="white", fontsize=10)
    if title:  ax.set_title(title,  color="white", fontsize=11, fontweight="bold", pad=10)

def _save(fig, path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    Saved: {path}")

def _legend(ax, algos):
    ax.legend(
        handles=[mpatches.Patch(color=algo_color(a), label=a) for a in algos],
        framealpha=0.25, labelcolor="white", facecolor="#1a1a2e",
        edgecolor="#444", fontsize=9)

def _sort_activities(acts):
    return sorted(acts, key=lambda a: (
        0 if a in WALKING_ACTIVITIES else
        1 if a in NON_WALKING_ACTIVITIES else 2, a))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Per-label avg steps/min per algorithm  (sensors averaged together)
# ─────────────────────────────────────────────────────────────────────────────

def plot1_label_avg_by_algorithm(df_long, out_dir):
    """
    For each PA_TYPE label, compute the mean steps/min averaged across all
    sensors, separately per algorithm. One grouped bar per activity.
    Walking and non-walking shown on the same chart with a divider.
    """
    algos = sorted(df_long["algorithm"].unique())

    # True activity rate: sum all steps / sum all duration across sensors & segments
    # (avoids many near-zero segments dragging the mean down)
    agg_raw = (df_long
               .groupby(["PA_TYPE", "activity_category", "algorithm"])
               .agg(total_steps=("total_steps","sum"), total_dur=("duration_sec","sum"))
               .reset_index())
    agg_raw["mean_spm"] = agg_raw["total_steps"] / agg_raw["total_dur"] * 60
    agg = agg_raw

    activities = _sort_activities(agg["PA_TYPE"].unique().tolist())
    # Drop activities with no data for any algo
    activities = [a for a in activities
                  if agg[agg["PA_TYPE"] == a].shape[0] > 0]

    n_algos = len(algos)
    y_pos   = np.arange(len(activities))
    bar_h   = 0.8 / n_algos
    offsets = np.linspace(-(n_algos-1)/2, (n_algos-1)/2, n_algos) * bar_h

    fig, ax = plt.subplots(figsize=(13, max(7, len(activities)*0.55 + 2)))
    fig.patch.set_facecolor(DARK_BG)

    for i, algo in enumerate(algos):
        sub  = agg[agg["algorithm"] == algo].set_index("PA_TYPE")
        vals = [sub.loc[a, "mean_spm"] if a in sub.index else 0 for a in activities]
        ax.barh(y_pos + offsets[i], vals, height=bar_h*0.9,
                color=algo_color(algo), alpha=0.88, label=algo)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([a.replace("_", " ") for a in activities],
                       fontsize=9, color="white")
    _style(ax, xlabel="Mean Steps / Min  (avg across all sensors)",
           title="Average Steps/Min per Activity Label — by Algorithm\n"
                 "(sensors averaged together)")

    # Walking / non-walking divider
    wc = sum(1 for a in activities if a in WALKING_ACTIVITIES)
    if 0 < wc < len(activities):
        ax.axhline(wc - 0.5, color="#666", linewidth=1.2, linestyle=":")
        ax.text(ax.get_xlim()[1]*0.98, wc - 0.15,
                "Walking ↑  |  Non-Walking ↓",
                color="#888", fontsize=8, ha="right")

    ax.spines[["top","right","left"]].set_visible(False)
    _legend(ax, algos)
    _save(fig, os.path.join(out_dir, "plot1_label_avg_by_algorithm.png"))

    # Also save the numbers
    agg_wide = agg.pivot_table(
        index=["PA_TYPE","activity_category"], columns="algorithm",
        values="mean_spm").round(2).reset_index()
    agg_wide.to_csv(os.path.join(out_dir, "plot1_label_avg_by_algorithm.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Per-sensor: FP-rate heatmap + best-algorithm highlight
# ─────────────────────────────────────────────────────────────────────────────

def plot2_sensor_algorithm_heatmap(df_long, out_dir):
    """
    Two panels:
      Left  — Heatmap: rows=sensors, cols=algorithms, value=FP rate (%)
              Cell with lowest FP rate per sensor is highlighted.
      Right — Bar chart: FP rate per sensor × algorithm, best algo marked.
    """
    algos   = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())

    nw = df_long[df_long["activity_category"] == "Non-Walking"].copy()
    if nw.empty:
        print("  [SKIP plot2] No non-walking segments found")
        return

    nw["detected"] = (nw["total_steps"] > 0).astype(float)
    fp = (nw.groupby(["sensor","algorithm"])["detected"]
            .mean()
            .mul(100)
            .reset_index(name="fp_pct"))

    # Build matrix
    mat = pd.DataFrame(index=sensors, columns=algos, dtype=float)
    for _, row in fp.iterrows():
        mat.loc[row["sensor"], row["algorithm"]] = row["fp_pct"]

    # Best algorithm per sensor = lowest FP rate
    best = mat.idxmin(axis=1)  # Series: sensor -> best_algo

    # ── figure ──────────────────────────────────────────────────────────────
    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(14, max(5, len(sensors)*0.9 + 2)),
        gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor(DARK_BG)

    # Left: heatmap
    cmap = LinearSegmentedColormap.from_list(
        "fp", ["#57CC99", "#F4A261", "#E84855"])   # green=low, red=high
    mv = mat.values.astype(float)
    im = ax_heat.imshow(mv, aspect="auto", cmap=cmap,
                        vmin=0, vmax=100, interpolation="nearest")
    ax_heat.set_facecolor(DARK_AX)
    ax_heat.set_xticks(range(len(algos)))
    ax_heat.set_xticklabels(algos, color="white", fontsize=10)
    ax_heat.set_yticks(range(len(sensors)))
    ax_heat.set_yticklabels(sensors, color="white", fontsize=9)
    ax_heat.tick_params(colors="white")

    for r, sensor in enumerate(sensors):
        for c, algo in enumerate(algos):
            v = mv[r, c]
            if not np.isnan(v):
                is_best = (algo == best[sensor])
                txt = ax_heat.text(c, r, f"{v:.1f}%", ha="center", va="center",
                                   fontsize=9, fontweight="bold",
                                   color="white" if v > 50 else "#111")
                if is_best:
                    ax_heat.add_patch(plt.Rectangle(
                        (c-0.48, r-0.48), 0.96, 0.96,
                        fill=False, edgecolor="white", linewidth=2.5))

    cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label("False Positive Rate (%)", color="white", fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax_heat.set_title("FP Rate per Sensor × Algorithm\n(★ = best per sensor)",
                      color="white", fontsize=11, fontweight="bold", pad=10)
    for sp in ax_heat.spines.values(): sp.set_color("#333")

    # Right: grouped bar — FP rate per sensor × algorithm
    n_sensors = len(sensors)
    x_pos     = np.arange(n_sensors)
    bw        = 0.8 / len(algos)
    offsets   = np.linspace(-(len(algos)-1)/2, (len(algos)-1)/2, len(algos)) * bw
    ax_bar.set_facecolor(DARK_AX)

    for i, algo in enumerate(algos):
        vals = [mat.loc[s, algo] if algo in mat.columns and not np.isnan(mat.loc[s, algo])
                else 0 for s in sensors]
        bars = ax_bar.bar(x_pos + offsets[i], vals, width=bw*0.9,
                          color=algo_color(algo), alpha=0.88, label=algo)
        # Star on best bar for each sensor
        for xi, (s, v) in enumerate(zip(sensors, vals)):
            if best[s] == algo:
                ax_bar.text(xi + offsets[i], v + 1.5, "★",
                            ha="center", color="white", fontsize=11)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(sensors, color="white", fontsize=9, rotation=20, ha="right")
    ax_bar.set_ylabel("False Positive Rate (%)", color="white", fontsize=10)
    ax_bar.set_ylim(0, 110)
    _style(ax_bar, title="Best Algorithm per Sensor\n(★ = lowest FP rate)")
    ax_bar.spines[["top","right"]].set_visible(False)
    _legend(ax_bar, algos)

    fig.suptitle("Sensor × Algorithm: False Positive Rate Analysis",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    _save(fig, os.path.join(out_dir, "plot2_sensor_algorithm_fp_heatmap.png"))

    # Save table with best-algo column
    mat_out = mat.copy().round(1)
    mat_out["best_algorithm"] = best
    mat_out.index.name = "sensor"
    mat_out.to_csv(os.path.join(out_dir, "plot2_sensor_fp_rates.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Per-DS pairwise algorithm difference + all-DS summary
# ─────────────────────────────────────────────────────────────────────────────

def plot3_pairwise_diff_per_ds(df_long, out_dir):
    """
    For each DS:
      - Average steps/min across sensors per algorithm per activity label
      - Compute pairwise differences between all 3 algorithm pairs
      - Show as grouped bar chart (one group per activity, one bar per pair)

    All-DS summary:
      - Box plot of pairwise differences across all DS and activities
    """
    algos  = sorted(df_long["algorithm"].unique())
    ds_ids = sorted(df_long["ds_id"].unique())

    if len(algos) < 2:
        print("  [SKIP plot3] Need ≥2 algorithms"); return

    pairs = [(algos[i], algos[j])
             for i in range(len(algos)) for j in range(i+1, len(algos))]
    pair_labels = [f"{a} − {b}" for a, b in pairs]
    pair_colors = ["#F4A261", "#9B5DE5", "#F15BB5"][:len(pairs)]

    all_ds_diffs = []   # collect for summary plot

    # ── Per-DS plots ──────────────────────────────────────────────────────────
    for ds_id in ds_ids:
        sub = df_long[df_long["ds_id"] == ds_id]

        # True activity rate: sum steps / sum duration across all sensors & segments
        # This avoids the many-zeros problem that makes mean_spm near zero
        agg_raw = (sub.groupby(["PA_TYPE","activity_category","algorithm"])
                      .agg(total_steps=("total_steps","sum"),
                           total_dur=("duration_sec","sum"))
                      .reset_index())
        agg_raw["mean_spm"] = agg_raw["total_steps"] / agg_raw["total_dur"] * 60

        activities = _sort_activities(agg_raw["PA_TYPE"].unique().tolist())
        # Use single index (PA_TYPE only) so reindex works correctly
        wide = agg_raw.pivot_table(
            index="PA_TYPE", columns="algorithm", values="mean_spm").reindex(activities)
        # keep agg alias for downstream collect step
        agg = agg_raw

        # Compute pairwise diffs
        diff_data = {}
        for (a, b), label in zip(pairs, pair_labels):
            if a in wide.columns and b in wide.columns:
                diff_data[label] = (wide[a] - wide[b]).values
            else:
                diff_data[label] = np.full(len(activities), np.nan)

        # Collect for all-DS summary
        act_cat_map = agg_raw.set_index("PA_TYPE")["activity_category"].to_dict()
        for label, vals in diff_data.items():
            for act, v in zip(activities, vals):
                if not np.isnan(v):
                    all_ds_diffs.append({
                        "ds_id": ds_id, "pair": label,
                        "PA_TYPE": act,
                        "activity_category": act_cat_map.get(act, "Other"),
                        "diff_spm": v})

        # ── Draw per-DS bar chart ─────────────────────────────────────────────
        n_acts  = len(activities)
        n_pairs = len(pairs)
        x_pos   = np.arange(n_acts)
        bw      = 0.8 / n_pairs
        offsets = np.linspace(-(n_pairs-1)/2, (n_pairs-1)/2, n_pairs) * bw

        fig, ax = plt.subplots(figsize=(max(12, n_acts*0.7 + 2), 6))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_AX)

        for i, (label, color) in enumerate(zip(pair_labels, pair_colors)):
            vals = diff_data[label]
            ax.bar(x_pos + offsets[i], vals, width=bw*0.9,
                   color=color, alpha=0.85, label=label)

        ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)

        # Walking / non-walking divider
        wc = sum(1 for a in activities if a in WALKING_ACTIVITIES)
        if 0 < wc < n_acts:
            ax.axvline(wc - 0.5, color="#666", linewidth=1.2, linestyle=":")
            ax.text(wc - 0.4, ax.get_ylim()[1]*0.92,
                    "Walking ←  |  Non-Walking →",
                    color="#888", fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([a.replace("_"," ") for a in activities],
                           color="white", fontsize=8, rotation=35, ha="right")
        ax.set_ylabel("Difference in Steps/Min  (A − B)", color="white", fontsize=10)
        _style(ax, title=f"{ds_id} — Pairwise Algorithm Difference per Activity\n"
                          "(sensors averaged; positive = first algo higher)")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(handles=[mpatches.Patch(color=c, label=l)
                            for l, c in zip(pair_labels, pair_colors)],
                  framealpha=0.25, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="#444", fontsize=9)
        _save(fig, os.path.join(out_dir, f"plot3_{ds_id}_pairwise_diff.png"))

    # ── All-DS summary: box plot + CSV of pairwise diffs ────────────────────
    if not all_ds_diffs:
        print("  [SKIP] No diff data collected for summary")
        return
    df_diff = pd.DataFrame(all_ds_diffs)

    # Save CSV
    df_diff.to_csv(os.path.join(out_dir, "plot3_all_pairwise_diffs.csv"), index=False)

    # One figure: left=walking, right=non-walking, rows=pairs
    cats = ["Walking", "Non-Walking"]
    fig, axes = plt.subplots(len(pairs), 2,
                             figsize=(14, len(pairs)*3.5 + 1),
                             sharey=False)
    fig.patch.set_facecolor(DARK_BG)
    if len(pairs) == 1: axes = [axes]

    for row_i, (label, color) in enumerate(zip(pair_labels, pair_colors)):
        for col_i, cat in enumerate(cats):
            ax = axes[row_i][col_i]
            ax.set_facecolor(DARK_AX)

            sub_diff = df_diff[(df_diff["pair"]==label) &
                                (df_diff["activity_category"]==cat)]
            acts_in  = _sort_activities(sub_diff["PA_TYPE"].unique().tolist())
            data     = [sub_diff[sub_diff["PA_TYPE"]==a]["diff_spm"].dropna().values
                        for a in acts_in]

            if not data or all(len(d)==0 for d in data):
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        color="#888", ha="center", va="center")
                continue

            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops={"color":"white","linewidth":2},
                            whiskerprops={"color":"#888"}, capprops={"color":"#888"},
                            flierprops={"marker":"o","markersize":3,
                                        "markerfacecolor":"#555","linestyle":"none"})
            for patch in bp["boxes"]:
                patch.set_facecolor(color); patch.set_alpha(0.75)

            ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xticks(range(1, len(acts_in)+1))
            ax.set_xticklabels([a.replace("_"," ") for a in acts_in],
                               color="white", fontsize=7, rotation=30, ha="right")
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_color("#333")
            ax.yaxis.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")
            ax.set_axisbelow(True)

            if col_i == 0:
                ax.set_ylabel(f"{label}\nDiff (steps/min)", color=color,
                              fontsize=9, fontweight="bold")
            if row_i == 0:
                ax.set_title(cat, color="white", fontsize=11, fontweight="bold")

    fig.suptitle("All DS — Pairwise Algorithm Differences by Activity\n"
                 "(each box = spread across all DS; 0 = algorithms agree)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    _save(fig, os.path.join(out_dir, "plot3_all_DS_summary_pairwise_diff.png"))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _process_ds_worker(args):
    """Top-level wrapper so multiprocessing can pickle the call."""
    ds_root, ds_id, algorithms, label_root = args
    return process_ds(ds_root, ds_id, algorithms, label_root)


def process_ds(ds_root, ds_id, algorithms, label_root=None):
    print(f"\n  ┌─ {ds_id}")
    label_path = find_label_file(ds_root, ds_id, label_root)
    if label_path is None:
        print(f"  └─ [SKIP] No label file found. Looked in:")
        if label_root:
            print(f"           {os.path.join(label_root, ds_id, 'label')}")
        print(f"           {ds_root}")
        return []
    print(f"  │  Labels: {label_path}")
    labels = load_labels(label_path)
    print(f"  │  {len(labels)} segments after filtering")

    records = []
    for algo in algorithms:
        sensor_files = find_steps_files(ds_root, ds_id, algo)
        if not sensor_files:
            print(f"  │  [SKIP] No steps files for algo={algo}"); continue

        # Deduplicate: same sensor name under same algo -> keep only first path
        seen_sensors = {}
        for sensor, fpath in sensor_files:
            if sensor not in seen_sensors:
                seen_sensors[sensor] = fpath
            else:
                print(f"  │  [DUP]  {algo:12s}  {sensor:22s}  skipping: {fpath}")

        for sensor, fpath in seen_sensors.items():
            try:
                series = load_steps_series(fpath)
                recs   = aggregate_segments(labels, series, ds_id, algo, sensor)
                records.extend(recs)
                print(f"  │  ✓ {algo:12s}  {sensor:22s}  "
                      f"{len(recs)} segs  sum={series.sum():.0f} steps")
            except Exception as e:
                print(f"  │  [ERROR] {algo}/{sensor}: {e}")

    print(f"  └─ {len(records)} records"); return records


def main():
    parser = argparse.ArgumentParser(description="Step algorithm comparison")
    parser.add_argument("--root",   default=DEFAULT_ROOT,
                        help="Root folder containing DS_* result subfolders")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help="Root folder containing DS_* label subfolders")
    parser.add_argument("--ds",     nargs="*", default=None,
                        help="Specific DS IDs e.g. DS_10 DS_12  (default: all)")
    parser.add_argument("--algos",  nargs="*", default=ALGORITHMS,
                        help="Algorithm subfolder names")
    parser.add_argument("--out",    default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for DS processing (default: 4)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("╔══════════════════════════════════════════════╗")
    print("║   STEP ALGORITHM COMPARISON — CLUSTER RUN   ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Root  : {args.root}")
    print(f"  Labels: {args.labels}")
    print(f"  Algos : {args.algos}")
    print(f"  Workers: {args.workers}")
    print(f"  Out   : {args.out}")

    all_ds = discover_ds_folders(args.root)
    ds_folders = [d for d in all_ds if os.path.basename(d) in args.ds] \
                 if args.ds else all_ds
    print(f"\n  {len(ds_folders)} DS folder(s): "
          f"{[os.path.basename(d) for d in ds_folders]}")

    # ── Parallel processing: one worker per DS ───────────────────────────────
    n_workers = min(len(ds_folders), args.workers)
    worker_args = [
        (ds_root, os.path.basename(ds_root), args.algos, args.labels)
        for ds_root in ds_folders
    ]

    all_records = []
    if n_workers > 1:
        print(f"\n  Using {n_workers} parallel workers...")
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_process_ds_worker, worker_args)
        for recs in results:
            all_records.extend(recs)
    else:
        for wa in worker_args:
            all_records.extend(_process_ds_worker(wa))

    if not all_records:
        print("\nERROR: No data collected. Check paths and algo names."); return

    df_long = pd.DataFrame(all_records)
    df_long.to_csv(os.path.join(args.out, "segments_long.csv"), index=False)
    print(f"\n  {len(df_long)} total segment records collected")

    print("\n  Generating plots...")
    print("\n  [1/3] Label avg steps/min per algorithm (sensors merged)...")
    plot1_label_avg_by_algorithm(df_long, args.out)

    print("\n  [2/3] Sensor × algorithm FP-rate heatmap...")
    plot2_sensor_algorithm_heatmap(df_long, args.out)

    print("\n  [3/3] Per-DS pairwise algorithm differences + all-DS summary...")
    plot3_pairwise_diff_per_ds(df_long, args.out)

    print(f"\n  ✓ Done! Output in: {args.out}")
    print()
    for f in sorted(os.listdir(args.out)):
        kb = os.path.getsize(os.path.join(args.out, f)) / 1024
        print(f"    {f:<55} {kb:6.1f} KB")


if __name__ == "__main__":
    main()