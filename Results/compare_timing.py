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
branch_name = "hip_alltogether"

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



# 1. approval log ===

# small, consistent fonts (follow your rcParams)
FS_BASE   = plt.rcParams['font.size']      # e.g. 8 from your rcParams block
FS_TOP    = FS_BASE                        # numbers on bar tops
TOP_DY    = 1                              # points offset for top labels
EPS       = 1e-6                           # for log-scale safety

# --- NEW: column for the Unified/No Prefetch series ---
COL_U_NP = "Time_NoPrefetch"   # change here if your column name differs

def fmt_num(v):
    """Show no decimals if value > 10, else 2 decimals."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    return f"{int(round(v))}" if v > 10 else f"{v:.2f}"

def y_at_std_top(mean, std):
    """Return y for label at top of ±1σ (mean+std). Falls back to mean if std missing."""
    if np.isfinite(mean) and np.isfinite(std):
        return max(mean + std, mean + EPS)  # ensure > mean (nice on log scale)
    return mean

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

# These correspond to SubmissionX / ExecutionX rows
STACK_OPS  = [("CopyToDevice", "CopyToDevice/Prefetch"),
              ("MakeClusters", "Kernel"),
              ("CopyToHost",   "CopyToHost/Prefetch")]

# Build unified group order: first 3 simple, then stacked trio, then remaining simple
groups = []
for op in SIMPLE_OPS[:3]: groups.append(("simple", op, op))
for key, pretty in STACK_OPS: groups.append(("stack", key, pretty))
for op in SIMPLE_OPS[3:]: groups.append(("simple", op, op))

# --- Filter groups: keep only ops with at least one valid value ---
valid_groups = []
for kind, key, pretty in groups:
    if kind == "simple":
        c_val = get_mean(key, "Time_Classic")
        u_val = get_mean(key, "Time_Unified")
        n_val = get_mean(key, COL_U_NP)
        if np.isfinite(c_val) or np.isfinite(u_val) or np.isfinite(n_val):
            valid_groups.append((kind, key, pretty))
    else:  # stacked ops
        c_exe = get_mean(f"Execution{key}", "Time_Classic")
        u_exe = get_mean(f"Execution{key}", "Time_Unified")
        n_exe = get_mean(f"Execution{key}", COL_U_NP)
        if np.isfinite(c_exe) or np.isfinite(u_exe) or np.isfinite(n_exe):
            valid_groups.append((kind, key, pretty))

groups = valid_groups

x = np.arange(len(groups))

# --- CHANGED: three bars per group (Classic, Unified, NoPrefetch) ---
bar_width = 0.22
OFFSETS = (-bar_width, 0.0, +bar_width)  # Classic, Unified, NoPrefetch

plt.figure(figsize=(14, 8), constrained_layout=True)

# Colors + legend labels (use your palette)
classic_exec = "#5790fc"; classic_sub  = "#2b477d"
unified_exec = "#f89c20"; unified_sub  = "#7a4d10"
nopref_exec  = "#e42536"; nopref_sub   = "#780f1c"

lbl_c_sub = "Classic-submission"
lbl_c_exe = "Classic-execution"
lbl_u_sub = "Unified-submission"
lbl_u_exe = "Unified-execution"
lbl_n_sub = "NoPrefetch-submission"
lbl_n_exe = "NoPrefetch-execution"

# whisker colors
c_err = "#2f5597"; u_err = "#a15c00"; n_err = "#b31d2b"

used_labels = set()
def add_bar(*args, label=None, **kw):
    if label and label in used_labels:
        label = "_nolegend_"
    elif label:
        used_labels.add(label)
    return plt.bar(*args, label=label, **kw)

def add_errbar(xc, y, s, color):
    """Draw ±1σ error bar, safe for log-scale (no negative lower bound)."""
    if np.isfinite(y) and np.isfinite(s) and y > EPS and s > 0:
        low = min(s, y - EPS)                      # cap so y - low > 0 on log scale
        yerr = np.array([[low], [s]])              # asymmetric: [lower; upper]
        plt.errorbar(xc, y, yerr=yerr, fmt="none",
                     ecolor=color, elinewidth=1.0, capsize=4, zorder=3)

# Draw bars
for i, (kind, key, pretty) in enumerate(groups):
    xc = x[i] + OFFSETS[0]  # classic
    xu = x[i] + OFFSETS[1]  # unified
    xn = x[i] + OFFSETS[2]  # no prefetch

    if kind == "simple":
        # simple ops: one value per mode (treated as 'execution' bars)
        c_val = get_mean(key, "Time_Classic")
        u_val = get_mean(key, "Time_Unified")
        n_val = get_mean(key, COL_U_NP)

        c_std = get_std(key,  "Time_Classic")
        u_std = get_std(key,  "Time_Unified")
        n_std = get_std(key,  COL_U_NP)

        add_bar(xc, [c_val], width=bar_width, color=classic_exec, label=lbl_c_exe)
        add_bar(xu, [u_val], width=bar_width, color=unified_exec, label=lbl_u_exe)
        add_bar(xn, [n_val], width=bar_width, color=nopref_exec, label=lbl_n_exe)

        # labels on top (mean only)
        if np.isfinite(c_val) and c_val > EPS:
            plt.annotate(fmt_num(c_val), xy=(xc, y_at_std_top(c_val, c_std)), xytext=(0, TOP_DY),
                         textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(u_val) and u_val > EPS:
            plt.annotate(fmt_num(u_val), xy=(xu, y_at_std_top(u_val, u_std)), xytext=(0, TOP_DY),
                         textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(n_val) and n_val > EPS:
            plt.annotate(fmt_num(n_val), xy=(xn, y_at_std_top(n_val, n_std)), xytext=(0, TOP_DY),
                         textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

        # std as error bars at the bar top
        add_errbar(xc, c_val, c_std, color=c_err)
        add_errbar(xu, u_val, u_std, color=u_err)
        add_errbar(xn, n_val, n_std, color=n_err)

    else:  # stacked GPU ops
        # rows are SubmissionX / ExecutionX (X = key)
        # Classic
        c_sub = get_mean(f"Submission{key}", "Time_Classic")
        c_exe = get_mean(f"Execution{key}",  "Time_Classic")
        c_exe_std = get_std(f"Execution{key}", "Time_Classic")
        c_top = max(c_exe - c_sub, 0.0) if np.isfinite(c_exe) and np.isfinite(c_sub) else np.nan

        # Unified
        u_sub = get_mean(f"Submission{key}", "Time_Unified")
        u_exe = get_mean(f"Execution{key}",  "Time_Unified")
        u_exe_std = get_std(f"Execution{key}", "Time_Unified")
        u_top = max(u_exe - u_sub, 0.0) if np.isfinite(u_exe) and np.isfinite(u_sub) else np.nan

        # No Prefetch (same op rows, different column)
        n_sub = get_mean(f"Submission{key}", COL_U_NP)
        n_exe = get_mean(f"Execution{key}",  COL_U_NP)
        n_exe_std = get_std(f"Execution{key}", COL_U_NP)
        n_top = max(n_exe - n_sub, 0.0) if np.isfinite(n_exe) and np.isfinite(n_sub) else np.nan

        # Classic stack
        add_bar(xc, [c_sub], width=bar_width, color=classic_sub,  label=lbl_c_sub)
        add_bar(xc, [c_top], width=bar_width, bottom=[c_sub], color=classic_exec, label=lbl_c_exe)

        # Unified stack
        add_bar(xu, [u_sub], width=bar_width, color=unified_sub,  label=lbl_u_sub)
        add_bar(xu, [u_top], width=bar_width, bottom=[u_sub], color=unified_exec, label=lbl_u_exe)

        # No Prefetch stack
        add_bar(xn, [n_sub], width=bar_width, color=nopref_sub,   label=lbl_n_sub)
        add_bar(xn, [n_top], width=bar_width, bottom=[n_sub], color=nopref_exec, label=lbl_n_exe)

        # ±1σ error bars at the EXECUTION total
        add_errbar(xc, c_exe, c_exe_std, color=c_err)
        add_errbar(xu, u_exe, u_exe_std, color=u_err)
        add_errbar(xn, n_exe, n_exe_std, color=n_err)

        # mean labels (Execution) at the top of ±1σ
        if np.isfinite(c_exe) and c_exe > EPS:
            plt.annotate(fmt_num(c_exe), xy=(xc, y_at_std_top(c_exe, c_exe_std)), xytext=(0, TOP_DY),
                         textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(u_exe) and u_exe > EPS:
            plt.annotate(fmt_num(u_exe), xy=(xu, y_at_std_top(u_exe, u_exe_std)), xytext=(0, TOP_DY),
                         textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(n_exe) and n_exe > EPS:
            plt.annotate(fmt_num(n_exe), xy=(xn, y_at_std_top(n_exe, n_exe_std)), xytext=(0, TOP_DY),
                         textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

# Axes/legend
labels = [pretty for _,_,pretty in groups]
plt.xticks(x, labels, rotation=15, ha="right", rotation_mode="anchor")
plt.ylabel("Time (ms)")
plt.yscale('log')  # logarithmic scale
plt.title("Classic vs Unified (Prefetch / No Prefetch) — Log scale")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Add a proxy legend entry for error bars (avoid duplicate whiskers)
from matplotlib.lines import Line2D
err_proxy = Line2D([0], [0], color="#444", lw=1, label="±1σ (std)")
handles, lbls = plt.gca().get_legend_handles_labels()
if "±1σ (std)" not in lbls:
    handles.append(err_proxy); lbls.append("±1σ (std)")
plt.legend(handles, lbls, loc="upper left", bbox_to_anchor=(1, 1))

plt.savefig(f"{results}/approval_log.png", dpi=300, bbox_inches="tight")
plt.show()

# 1.2 approval linear ===

# small, consistent fonts (follow your rcParams)
FS_BASE   = plt.rcParams['font.size']      # e.g. 8 from your rcParams block
FS_TOP    = FS_BASE                        # numbers on bar tops
TOP_DY    = 1                              # points offset for top labels
EPS       = 1e-6                           # for log-scale safety

def fmt_pm(mean, std):
    if np.isfinite(mean) and np.isfinite(std):
        return f"{mean:.2f} ± {std:.2f}"
    elif np.isfinite(mean):
        return f"{mean:.2f}"
    return ""

def y_at_std_top(mean, std):
    """Return y for label at top of ±1σ (mean+std). Falls back to mean if std missing."""
    if np.isfinite(mean) and np.isfinite(std):
        return max(mean + std, mean + EPS)  # ensure > mean (nice on log scale)
    return mean

# Clean strings and index once
merged_mean["Operation"] = merged_mean["Operation"].astype(str).str.strip()
df_idx  = merged_mean.set_index("Operation")
std_idx = merged_std.set_index("Operation")

def get_mean(op, col):
    return float(df_idx.at[op, col]) if op in df_idx.index else np.nan

def get_std(op, col):
    return float(std_idx.at[op, col]) if op in std_idx.index else np.nan

# X-axis order (your list)
SIMPLE_OPS = ["readDataFromFile", "allocateInputData", "allocateOutputData",
              "writeDataToFile", "freeInputData", "freeOutputData"]

STACK_OPS  = [("CopyToDevice", "CopyToDevice/Prefetch"),
              ("MakeClusters", "Kernel"),
              ("CopyToHost",   "CopyToHost/Prefetch")]

# Build unified group order: first 3 simple, then stacked trio, then remaining simple
groups = []
for op in SIMPLE_OPS[:3]: groups.append(("simple", op, op))
for key, pretty in STACK_OPS: groups.append(("stack", key, pretty))
for op in SIMPLE_OPS[3:]: groups.append(("simple", op, op))

x = np.arange(len(groups))
bar_width = 0.35

plt.figure(figsize=(14, 8), constrained_layout=True)

# Colors + legend labels (your palette)
classic_exec = "#5790fc"; classic_sub  = "#2b477d"
unified_exec = "#f89c20"; unified_sub  = "#7a4d10"

lbl_c_sub = "Classic-submission"
lbl_c_exe = "Classic-execution"
lbl_u_sub = "Unified-submission"
lbl_u_exe = "Unified-execution"
used_labels = set()

def add_bar(*args, label=None, **kw):
    if label and label in used_labels:
        label = "_nolegend_"
    elif label:
        used_labels.add(label)
    return plt.bar(*args, label=label, **kw)

def add_errbar(xc, y, s, color):
    """Draw ±1σ error bar, safe for log-scale (no negative lower bound)."""
    if np.isfinite(y) and np.isfinite(s) and y > EPS and s > 0:
        low = min(s, y - EPS)                      # cap so y - low > 0 on log scale
        yerr = np.array([[low], [s]])              # asymmetric: [lower; upper]
        plt.errorbar(xc, y, yerr=yerr, fmt="none",
                     ecolor=color, elinewidth=1.0, capsize=4, zorder=3)

# Draw bars
for i, (kind, key, pretty) in enumerate(groups):
    if kind == "simple":
        # simple ops: one value per mode (treated as 'execution' bars)
        c_val = get_mean(key, "Time_Classic")
        u_val = get_mean(key, "Time_Unified")
        c_std = get_std(key,  "Time_Classic")
        u_std = get_std(key,  "Time_Unified")

        xc = x[i] - bar_width/2
        xu = x[i] + bar_width/2

        add_bar(xc, [c_val], width=bar_width, color=classic_exec, label=lbl_c_exe)
        add_bar(xu, [u_val], width=bar_width, color=unified_exec, label=lbl_u_exe)

        # labels on top
        # mean labels at the TOP of the ±1σ whisker (no ± text)
        if np.isfinite(c_val) and c_val > EPS:
            ylab_c = y_at_std_top(c_val, c_std)   # -> mean + std (safe on log)
            plt.annotate(f"{c_val:.2f}", xy=(xc, ylab_c), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

        if np.isfinite(u_val) and u_val > EPS:
            ylab_u = y_at_std_top(u_val, u_std)
            plt.annotate(f"{u_val:.2f}", xy=(xu, ylab_u), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

        # std as error bars at the bar top
        add_errbar(xc, c_val, c_std, color="#2f5597")   # classic whisker color
        add_errbar(xu, u_val, u_std, color="#a15c00")   # unified whisker color

    else:  # stacked GPU ops
        # === Stacked GPU ops: compute means and stds (EXECUTION std!) ===
        c_sub = get_mean(f"Submission{key}", "Time_Classic")
        c_exe = get_mean(f"Execution{key}",  "Time_Classic")
        u_sub = get_mean(f"Submission{key}", "Time_Unified")
        u_exe = get_mean(f"Execution{key}",  "Time_Unified")

        # std for the TOTAL height (Execution), not Submission
        c_exe_std = get_std(f"Execution{key}", "Time_Classic")
        u_exe_std = get_std(f"Execution{key}", "Time_Unified")

        # top segment = Execution − Submission (never negative)
        c_top = max(c_exe - c_sub, 0.0) if np.isfinite(c_exe) and np.isfinite(c_sub) else np.nan
        u_top = max(u_exe - u_sub, 0.0) if np.isfinite(u_exe) and np.isfinite(u_sub) else np.nan

        xc = x[i] - bar_width/2
        xu = x[i] + bar_width/2

        # Classic stack
        add_bar(xc, [c_sub], width=bar_width, color=classic_sub,  label=lbl_c_sub)
        add_bar(xc, [c_top], width=bar_width, bottom=[c_sub], color=classic_exec, label=lbl_c_exe)

        # Unified stack
        add_bar(xu, [u_sub], width=bar_width, color=unified_sub,  label=lbl_u_sub)
        add_bar(xu, [u_top], width=bar_width, bottom=[u_sub], color=unified_exec, label=lbl_u_exe)

        # ±1σ error bars at the EXECUTION total
        add_errbar(xc, c_exe, c_exe_std, color="#2f5597")
        add_errbar(xu, u_exe, u_exe_std, color="#a15c00")

        # mean labels (Execution) at the TOP of the ±1σ whisker
        if np.isfinite(c_exe) and c_exe > EPS:
            ylab_c = y_at_std_top(c_exe, c_exe_std)
            plt.annotate(f"{c_exe:.2f}", xy=(xc, ylab_c), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

        if np.isfinite(u_exe) and u_exe > EPS:
            ylab_u = y_at_std_top(u_exe, u_exe_std)
            plt.annotate(f"{u_exe:.2f}", xy=(xu, ylab_u), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

# Axes/legend
labels = [pretty for _,_,pretty in groups]
plt.xticks(x, labels, rotation=15, ha="right", rotation_mode="anchor")
plt.ylabel("Time (ms)")
plt.title("Classic vs Unified Memory")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Add a proxy legend entry for error bars (avoid duplicate whiskers)
from matplotlib.lines import Line2D
err_proxy = Line2D([0], [0], color="#444", lw=1, label="±1σ (std)")
handles, labels = plt.gca().get_legend_handles_labels()
if "±1σ (std)" not in labels:
    handles.append(err_proxy); labels.append("±1σ (std)")
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

plt.savefig(f"{results}/approval_linear.png", dpi=300, bbox_inches="tight")
plt.show()

# 1.3 approval linear zoomed in ===

# small, consistent fonts (follow your rcParams)
FS_BASE   = plt.rcParams['font.size']      # e.g. 8 from your rcParams block
FS_TOP    = FS_BASE                        # numbers on bar tops
TOP_DY    = 1                              # points offset for top labels
EPS       = 1e-6                           # for log-scale safety

def fmt_pm(mean, std):
    if np.isfinite(mean) and np.isfinite(std):
        return f"{mean:.2f} ± {std:.2f}"
    elif np.isfinite(mean):
        return f"{mean:.2f}"
    return ""

def y_at_std_top(mean, std):
    """Return y for label at top of ±1σ (mean+std). Falls back to mean if std missing."""
    if np.isfinite(mean) and np.isfinite(std):
        return max(mean + std, mean + EPS)  # ensure > mean (nice on log scale)
    return mean

# Clean strings and index once
merged_mean["Operation"] = merged_mean["Operation"].astype(str).str.strip()
df_idx  = merged_mean.set_index("Operation")
std_idx = merged_std.set_index("Operation")

def get_mean(op, col):
    return float(df_idx.at[op, col]) if op in df_idx.index else np.nan

def get_std(op, col):
    return float(std_idx.at[op, col]) if op in std_idx.index else np.nan

# X-axis order (your list)
SIMPLE_OPS = ["readDataFromFile", "allocateInputData", "allocateOutputData",
              "writeDataToFile", "freeInputData", "freeOutputData"]

STACK_OPS  = [("CopyToDevice", "CopyToDevice/Prefetch"),
              ("MakeClusters", "Kernel"),
              ("CopyToHost",   "CopyToHost/Prefetch")]

# Build unified group order: first 3 simple, then stacked trio, then remaining simple
groups = []
for op in SIMPLE_OPS[:3]: groups.append(("simple", op, op))
for key, pretty in STACK_OPS: groups.append(("stack", key, pretty))
for op in SIMPLE_OPS[3:]: groups.append(("simple", op, op))

x = np.arange(len(groups))
bar_width = 0.35

plt.figure(figsize=(14, 8), constrained_layout=True)

# Colors + legend labels (your palette)
classic_exec = "#5790fc"; classic_sub  = "#2b477d"
unified_exec = "#f89c20"; unified_sub  = "#7a4d10"

lbl_c_sub = "Classic-submission"
lbl_c_exe = "Classic-execution"
lbl_u_sub = "Unified-submission"
lbl_u_exe = "Unified-execution"
used_labels = set()

def add_bar(*args, label=None, **kw):
    if label and label in used_labels:
        label = "_nolegend_"
    elif label:
        used_labels.add(label)
    return plt.bar(*args, label=label, **kw)

def add_errbar(xc, y, s, color):
    """Draw ±1σ error bar, safe for log-scale (no negative lower bound)."""
    if np.isfinite(y) and np.isfinite(s) and y > EPS and s > 0:
        low = min(s, y - EPS)                      # cap so y - low > 0 on log scale
        yerr = np.array([[low], [s]])              # asymmetric: [lower; upper]
        plt.errorbar(xc, y, yerr=yerr, fmt="none",
                     ecolor=color, elinewidth=1.0, capsize=4, zorder=3)

# Draw bars
for i, (kind, key, pretty) in enumerate(groups):
    if kind == "simple":
        # simple ops: one value per mode (treated as 'execution' bars)
        c_val = get_mean(key, "Time_Classic")
        u_val = get_mean(key, "Time_Unified")
        c_std = get_std(key,  "Time_Classic")
        u_std = get_std(key,  "Time_Unified")

        xc = x[i] - bar_width/2
        xu = x[i] + bar_width/2

        add_bar(xc, [c_val], width=bar_width, color=classic_exec, label=lbl_c_exe)
        add_bar(xu, [u_val], width=bar_width, color=unified_exec, label=lbl_u_exe)

        # labels on top
        # mean labels at the TOP of the ±1σ whisker (no ± text)
        if np.isfinite(c_val) and c_val > EPS:
            ylab_c = y_at_std_top(c_val, c_std)   # -> mean + std (safe on log)
            plt.annotate(f"{c_val:.2f}", xy=(xc, ylab_c), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

        if np.isfinite(u_val) and u_val > EPS:
            ylab_u = y_at_std_top(u_val, u_std)
            plt.annotate(f"{u_val:.2f}", xy=(xu, ylab_u), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

        # std as error bars at the bar top
        add_errbar(xc, c_val, c_std, color="#2f5597")   # classic whisker color
        add_errbar(xu, u_val, u_std, color="#a15c00")   # unified whisker color

    else:  # stacked GPU ops
        # === Stacked GPU ops: compute means and stds (EXECUTION std!) ===
        c_sub = get_mean(f"Submission{key}", "Time_Classic")
        c_exe = get_mean(f"Execution{key}",  "Time_Classic")
        u_sub = get_mean(f"Submission{key}", "Time_Unified")
        u_exe = get_mean(f"Execution{key}",  "Time_Unified")

        # std for the TOTAL height (Execution), not Submission
        c_exe_std = get_std(f"Execution{key}", "Time_Classic")
        u_exe_std = get_std(f"Execution{key}", "Time_Unified")

        # top segment = Execution − Submission (never negative)
        c_top = max(c_exe - c_sub, 0.0) if np.isfinite(c_exe) and np.isfinite(c_sub) else np.nan
        u_top = max(u_exe - u_sub, 0.0) if np.isfinite(u_exe) and np.isfinite(u_sub) else np.nan

        xc = x[i] - bar_width/2
        xu = x[i] + bar_width/2

        # Classic stack
        add_bar(xc, [c_sub], width=bar_width, color=classic_sub,  label=lbl_c_sub)
        add_bar(xc, [c_top], width=bar_width, bottom=[c_sub], color=classic_exec, label=lbl_c_exe)

        # Unified stack
        add_bar(xu, [u_sub], width=bar_width, color=unified_sub,  label=lbl_u_sub)
        add_bar(xu, [u_top], width=bar_width, bottom=[u_sub], color=unified_exec, label=lbl_u_exe)

        # ±1σ error bars at the EXECUTION total
        add_errbar(xc, c_exe, c_exe_std, color="#2f5597")
        add_errbar(xu, u_exe, u_exe_std, color="#a15c00")

        # mean labels (Execution) at the TOP of the ±1σ whisker
        if np.isfinite(c_exe) and c_exe > EPS:
            ylab_c = y_at_std_top(c_exe, c_exe_std)
            plt.annotate(f"{c_exe:.2f}", xy=(xc, ylab_c), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

        if np.isfinite(u_exe) and u_exe > EPS:
            ylab_u = y_at_std_top(u_exe, u_exe_std)
            plt.annotate(f"{u_exe:.2f}", xy=(xu, ylab_u), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

# Axes/legend
labels = [pretty for _,_,pretty in groups]
plt.xticks(x, labels, rotation=15, ha="right", rotation_mode="anchor")
plt.ylabel("Time (ms)")
plt.title("Classic vs Unified Memory")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Add a proxy legend entry for error bars (avoid duplicate whiskers)
from matplotlib.lines import Line2D
err_proxy = Line2D([0], [0], color="#444", lw=1, label="±1σ (std)")
handles, labels = plt.gca().get_legend_handles_labels()
if "±1σ (std)" not in labels:
    handles.append(err_proxy); labels.append("±1σ (std)")
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

# --- Also save a zoomed-in version (linear 0–10 ms) ---
ax = plt.gca()
ax.set_yscale('linear')
ax.set_ylim(0, 8)                         # pick 4, 6, 8, 10… as you like
ax.set_autoscale_on(False) 
ax.set_title("Classic vs Unified Memory — Zoomed 0–8 ms")

plt.savefig(f"{results}/approval_linear_zoom_in.png", dpi=300, bbox_inches="tight")

plt.show()
