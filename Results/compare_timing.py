#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
import os
import sys
import subprocess
import io
import math

warnings.filterwarnings("ignore", category=UserWarning)

results = 'Results'
if len(sys.argv) > 1:
    results = sys.argv[1]

def read_csv_from_branch(branch, filepath):
    """Read a CSV from another git branch (without checkout)."""
    blob = subprocess.check_output(["git", "show", f"{branch}:{filepath}"])
    return pd.read_csv(io.BytesIO(blob))

def normalize_operation(op: str) -> str:
    """Map all Kernel variants to one canonical naming (MakeClusters style)."""
    if op is None:
        return op
    s = str(op).strip()
    # Canonical: "SubmissionMakeClusters" / "ExecutionMakeClusters"
    if s == "kernelSubmission":
        return "SubmissionMakeClusters"
    if s == "kernelExecution":
        return "ExecutionMakeClusters"
    return s

# ---------------------------
# Classic (local files)
# ---------------------------
classic_files = sorted(glob.glob(f"{results}/results_classic*.csv"))
classic_dfs = [pd.read_csv(f) for f in classic_files if not f.endswith("results_classic0.csv")]

# Save the order from the first file (normalized)
classic_order_raw = classic_dfs[0]['Operation'].tolist()
classic_order = [normalize_operation(x) for x in classic_order_raw]

# Normalize op names BEFORE concat
for df in classic_dfs:
    df['Operation'] = df['Operation'].map(normalize_operation)

combined_c = pd.concat(classic_dfs, ignore_index=True)
combined_c['Operation'] = pd.Categorical(combined_c['Operation'], categories=classic_order, ordered=True)

mean_df_c = combined_c.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_c  = combined_c.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

mean_df_c.to_csv(f"{results}/classic_mean.csv", index=False)
std_df_c.to_csv(f"{results}/classic_std.csv", index=False)

# ---------------------------
# Unified (Prefetch) (local files)
# ---------------------------
unified_files = sorted(glob.glob(f"{results}/results_unified*.csv"))
unified_dfs = [pd.read_csv(f) for f in unified_files if not f.endswith("results_unified0.csv")]

# Normalize before concat
for df in unified_dfs:
    df['Operation'] = df['Operation'].map(normalize_operation)

combined_u = pd.concat(unified_dfs, ignore_index=True)
combined_u['Operation'] = pd.Categorical(combined_u['Operation'], categories=classic_order, ordered=True)

mean_df_u = combined_u.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_u  = combined_u.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

mean_df_u.to_csv(f"{results}/unified_mean.csv", index=False)
std_df_u.to_csv(f"{results}/unified_std.csv", index=False)

# ---------------------------
# Unified (No Prefetch) (from branch)
# ---------------------------
branch_name = "async_operations_alltogether"

unified_no_prefetch_dfs = []
missing = []
for i in range(1, 11):   # 1..10 (skip 0, like you had)
    path = f"{results}/results_unified_no_prefetch{i}.csv"
    try:
        df_np = read_csv_from_branch(branch_name, path)
        # Normalize immediately
        df_np['Operation'] = df_np['Operation'].map(normalize_operation)
        unified_no_prefetch_dfs.append(df_np)
    except subprocess.CalledProcessError:
        missing.append(path)

if not unified_no_prefetch_dfs:
    raise FileNotFoundError(
        f"No unified_no_prefetch files could be read from branch '{branch_name}'. "
        f"Tried indices 1..10 under {results}/."
    )
if missing:
    print(f"[warn] Missing from branch {branch_name}: {len(missing)} files. Example: {missing[0]}")

combined_u_no_prefetch = pd.concat(unified_no_prefetch_dfs, ignore_index=True)
combined_u_no_prefetch['Operation'] = pd.Categorical(
    combined_u_no_prefetch['Operation'], categories=classic_order, ordered=True
)

mean_df_u_no_prefetch = combined_u_no_prefetch.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_u_no_prefetch  = combined_u_no_prefetch.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

# ---------------------------
# Build merged tables (OUTER merges so nothing is dropped)
# ---------------------------
classic_mean = mean_df_c
classic_std  = std_df_c

unified_mean = mean_df_u
unified_std  = std_df_u

unified_no_prefetch_mean = mean_df_u_no_prefetch
unified_no_prefetch_std  = std_df_u_no_prefetch

merged_mean = classic_mean.merge(unified_mean, on='Operation', how='outer', suffixes=('_Classic', '_Unified'))
merged_mean = merged_mean.merge(
    unified_no_prefetch_mean[['Operation', 'Time']].rename(columns={'Time': 'Time_NoPrefetch'}),
    on='Operation',
    how='outer'
)

merged_std = classic_std.merge(unified_std, on='Operation', how='outer', suffixes=('_Classic', '_Unified'))
merged_std = merged_std.merge(
    unified_no_prefetch_std[['Operation', 'Time']].rename(columns={'Time': 'Time_NoPrefetch'}),
    on='Operation',
    how='outer'
)

# Save for plotting
merged_mean.to_csv(f"{results}/mean_timing_comparison.csv", index=False)


# === plotting (refactored to avoid repetition) ===

# small, consistent fonts (follow your rcParams)
FS_BASE   = plt.rcParams['font.size']      # e.g. 8 from your rcParams block
FS_TOP    = FS_BASE * 0.8                  # numbers on bar tops
TOP_DY    = 1                              # points offset for top labels
EPS       = 1e-6                           # for log-scale safety

COL_U_NP = "Time_NoPrefetch"   # column for Unified/No Prefetch

def fmt_num(v): 
    try: 
        v = float(v) 
    except (TypeError, ValueError): 
        return "" 
    if not np.isfinite(v): 
        return "" 
    return f"{int(round(v))}" if v > 10 else f"{v:.2f}"

def y_at_std_top(mean, std):
    if np.isfinite(mean) and np.isfinite(std):
        return max(mean + std, mean + EPS)
    return mean

def round_with_error(mean, std):
    """
    Apply rule on std: 1 significant digit (or 2 if first digit is 1).
    Round mean to the same decimal precision.
    Returns (mean_rounded, std_rounded).
    Handles zeros/NaNs/infs safely.
    """
    try:
        m = float(mean)
        s = float(std)
    except Exception:
        return mean, std

    if not (math.isfinite(m) and math.isfinite(s)) or s == 0:
        return mean, std

    exp = int(math.floor(math.log10(abs(s))))
    first_digit = int(abs(s) / (10 ** exp))

    sig_digits = 2 if first_digit == 1 else 1
    std_rounded = round(s, -exp + (sig_digits - 1))

    # number of decimals for mean, derived from rounded std
    if std_rounded == 0 or not math.isfinite(std_rounded):
        return mean, std_rounded
    decimals = max(-int(math.floor(math.log10(abs(std_rounded)))), 0)
    mean_rounded = round(m, decimals)
    return mean_rounded, std_rounded

def fmt_mean_with_error(mean, std):
    """
    Return a string with the mean rounded according to the std rule.
    If std is missing/zero, fall back to your previous fmt_num.
    """
    m_rounded, s_rounded = round_with_error(mean, std)
    # If we couldn't apply rule (std missing/zero), use your old fmt_num
    if m_rounded is mean and (not (isinstance(std, (int,float)) and math.isfinite(std)) or std == 0):
        return fmt_num(mean)
    # Format with the decimals implied by s_rounded
    if not (isinstance(s_rounded, (int,float)) and math.isfinite(s_rounded)) or s_rounded == 0:
        return fmt_num(m_rounded)
    # derive decimals from s_rounded
    dec = max(-int(math.floor(math.log10(abs(s_rounded)))) if s_rounded != 0 else 0, 0)
    return f"{m_rounded:.{dec}f}"

# Clean strings and index once
merged_mean["Operation"] = merged_mean["Operation"].astype(str).str.strip()
df_idx  = merged_mean.set_index("Operation")
std_idx = merged_std.set_index("Operation") if 'merged_std' in globals() else None

def get_mean(op, col):
    try:
        return float(df_idx.at[op, col])
    except Exception:
        return np.nan

def get_std(op, col):
    if std_idx is None:
        return np.nan
    try:
        return float(std_idx.at[op, col])
    except Exception:
        return np.nan

# X-axis order
SIMPLE_OPS = ["readDataFromFile", "allocateInputData", "allocateOutputData",
              "writeDataToFile", "freeInputData", "freeOutputData"]
STACK_OPS  = [("CopyToDevice", "CopyToDevice/Prefetch"),
              ("MakeClusters", "Kernel"),
              ("CopyToHost",   "CopyToHost/Prefetch")]

groups_all = []
for op in SIMPLE_OPS[:3]: groups_all.append(("simple", op, op))
for key, pretty in STACK_OPS: groups_all.append(("stack", key, pretty))
for op in SIMPLE_OPS[3:]: groups_all.append(("simple", op, op))

# Filter groups: keep only ops with at least one valid value
valid_groups = []
for kind, key, pretty in groups_all:
    if kind == "simple":
        if any(np.isfinite(v) for v in (
            get_mean(key, "Time_Classic"),
            get_mean(key, "Time_Unified"),
            get_mean(key, COL_U_NP),
        )):
            valid_groups.append((kind, key, pretty))
    else:
        if any(np.isfinite(v) for v in (
            get_mean(f"Execution{key}", "Time_Classic"),
            get_mean(f"Execution{key}", "Time_Unified"),
            get_mean(f"Execution{key}", COL_U_NP),
        )):
            valid_groups.append((kind, key, pretty))
groups = valid_groups

x = np.arange(len(groups))
bar_width = 0.22
OFFSETS = (-bar_width, 0.0, +bar_width)  # Classic, Unified, NoPrefetch

# colors & labels
classic_exec = "#5790fc"; classic_sub  = "#2b477d"
unified_exec = "#f89c20"; unified_sub  = "#7a4d10"
nopref_exec  = "#e42536"; nopref_sub   = "#780f1c"
c_err = "#2f5597"; u_err = "#a15c00"; n_err = "#b31d2b"

lbl_c_sub = "Classic-submission";   lbl_c_exe = "Classic-execution"
lbl_u_sub = "Unified-submission";   lbl_u_exe = "Unified-execution"
lbl_n_sub = "NoPrefetch-submission";lbl_n_exe = "NoPrefetch-execution"

def draw_plot(ax):
    used_labels = set()

    def add_bar(*args, label=None, **kw):
        if label and label in used_labels:
            label = "_nolegend_"
        elif label:
            used_labels.add(label)
        return ax.bar(*args, label=label, **kw)

    def add_errbar(xc, y, s, color):
        if np.isfinite(y) and np.isfinite(s) and y > EPS and s > 0:
            low = min(s, y - EPS)
            yerr = np.array([[low], [s]])
            ax.errorbar(xc, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.0, capsize=4, zorder=3)

    for i, (kind, key, pretty) in enumerate(groups):
        xc = x[i] + OFFSETS[0]  # classic
        xu = x[i] + OFFSETS[1]  # unified
        xn = x[i] + OFFSETS[2]  # no prefetch

        if kind == "simple":
            c_val = get_mean(key, "Time_Classic"); c_std = get_std(key, "Time_Classic")
            u_val = get_mean(key, "Time_Unified"); u_std = get_std(key, "Time_Unified")
            n_val = get_mean(key, COL_U_NP);       n_std = get_std(key, COL_U_NP)

            add_bar(xc, [c_val], width=bar_width, color=classic_exec, label=lbl_c_exe)
            add_bar(xu, [u_val], width=bar_width, color=unified_exec, label=lbl_u_exe)
            add_bar(xn, [n_val], width=bar_width, color=nopref_exec, label=lbl_n_exe)

            ax.annotate(fmt_mean_with_error(c_val, c_std), (xc, y_at_std_top(c_val, c_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
            ax.annotate(fmt_mean_with_error(u_val, u_std), (xu, y_at_std_top(u_val, u_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
            ax.annotate(fmt_mean_with_error(n_val, n_std), (xn, y_at_std_top(n_val, n_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

            add_errbar(xc, c_val, c_std, c_err)
            add_errbar(xu, u_val, u_std, u_err)
            add_errbar(xn, n_val, n_std, n_err)

        else:  # stacked
            c_sub = get_mean(f"Submission{key}", "Time_Classic")
            c_exe = get_mean(f"Execution{key}",  "Time_Classic"); c_exe_std = get_std(f"Execution{key}", "Time_Classic")
            c_top = max(c_exe - c_sub, 0.0) if np.isfinite(c_exe) and np.isfinite(c_sub) else np.nan

            u_sub = get_mean(f"Submission{key}", "Time_Unified")
            u_exe = get_mean(f"Execution{key}",  "Time_Unified"); u_exe_std = get_std(f"Execution{key}", "Time_Unified")
            u_top = max(u_exe - u_sub, 0.0) if np.isfinite(u_exe) and np.isfinite(u_sub) else np.nan

            n_sub = get_mean(f"Submission{key}", COL_U_NP)
            n_exe = get_mean(f"Execution{key}",  COL_U_NP); n_exe_std = get_std(f"Execution{key}", COL_U_NP)
            n_top = max(n_exe - n_sub, 0.0) if np.isfinite(n_exe) and np.isfinite(n_sub) else np.nan

            add_bar(xc, [c_sub], width=bar_width, color=classic_sub,  label=lbl_c_sub)
            add_bar(xc, [c_top], width=bar_width, bottom=[c_sub], color=classic_exec, label=lbl_c_exe)
            add_bar(xu, [u_sub], width=bar_width, color=unified_sub,  label=lbl_u_sub)
            add_bar(xu, [u_top], width=bar_width, bottom=[u_sub], color=unified_exec, label=lbl_u_exe)
            add_bar(xn, [n_sub], width=bar_width, color=nopref_sub,   label=lbl_n_sub)
            add_bar(xn, [n_top], width=bar_width, bottom=[n_sub], color=nopref_exec, label=lbl_n_exe)

            add_errbar(xc, c_exe, c_exe_std, c_err)
            add_errbar(xu, u_exe, u_exe_std, u_err)
            add_errbar(xn, n_exe, n_exe_std, n_err)

            ax.annotate(fmt_mean_with_error(c_exe, c_exe_std), (xc, y_at_std_top(c_exe, c_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
            ax.annotate(fmt_mean_with_error(u_exe, u_exe_std), (xu, y_at_std_top(u_exe, u_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
            ax.annotate(fmt_mean_with_error(n_exe, n_exe_std), (xn, y_at_std_top(n_exe, n_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

    labels = [pretty for _,_,pretty in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="center", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=10) 
    ax.set_ylabel("Time (ms)")
    ax.set_title("Classic vs Unified (Prefetch / No Prefetch)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    from matplotlib.lines import Line2D
    err_proxy = Line2D([0], [0], color="#444", lw=1, label="±1σ (std)")
    handles, lbls = ax.get_legend_handles_labels()
    if "±1σ (std)" not in lbls:
        handles.append(err_proxy); lbls.append("±1σ (std)")
    ax.legend(handles, lbls, loc="upper left", bbox_to_anchor=(1, 1))

def make_variant(scale="linear", ylim=None, suffix="linear"):
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    draw_plot(ax)
    ax.set_yscale(scale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    out = f"{results}/approval_{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

# --- one code path → three outputs ---
make_variant(scale="linear", ylim=None,   suffix="linear")
make_variant(scale="log",    ylim=None,   suffix="log")
make_variant(scale="linear", ylim=(0, 8), suffix="linear_zoom_in")
