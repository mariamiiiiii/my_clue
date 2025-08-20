import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Collect all results_classicX.csv files
classic_files = sorted(glob.glob("Results_MI300X/results_classic*.csv"))
classic_dfs = [pd.read_csv(f) for f in classic_files if not f.endswith("results_classic0.csv")]

# Save the order from the first file
classic_order = classic_dfs[0]['Operation'].tolist()

# Combine all runs into one DataFrame
combined_c = pd.concat(classic_dfs)

# Use Categorical to enforce the original order
combined_c['Operation'] = pd.Categorical(combined_c['Operation'], categories=classic_order, ordered=True)

# Group by Operation and compute mean and std
mean_df_c = combined_c.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_c = combined_c.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

# Save results to CSV
mean_df_c.to_csv("Results_MI300X/classic_mean.csv", index=False)
std_df_c.to_csv("Results_MI300X/classic_std.csv", index=False)


# Collect all results_unifiedX.csv files
unified_files = sorted(glob.glob("Results_MI300X/results_unified[1-9].csv") + glob.glob("Results_MI300X/results_unified10.csv")
)
unified_dfs = [pd.read_csv(f) for f in unified_files if not f.endswith("results_unified0.csv")]

# Combine all runs into one DataFrame
combined_u = pd.concat(unified_dfs)

# Use Categorical to enforce the original order
combined_u['Operation'] = pd.Categorical(combined_u['Operation'], categories=classic_order, ordered=True)

# Group by Operation and compute mean and std
mean_df_u = combined_u.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_u = combined_u.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

# Save results to CSV
mean_df_u.to_csv("Results_MI300X/unified_mean.csv", index=False)
std_df_u.to_csv("Results_MI300X/unified_std.csv", index=False)


# Collect all results_unified_no_prefetchX.csv files
unified_no_prefetch_files = sorted(glob.glob("Results_MI300X/results_unified_no_prefetch*.csv"))
unified_no_prefetch_dfs = [pd.read_csv(f) for f in unified_no_prefetch_files if not f.endswith("results_unified_no_prefetch0.csv")]

# Combine all runs into one DataFrame
combined_u_no_prefetch = pd.concat(unified_no_prefetch_dfs)

# Use Categorical to enforce the original order
combined_u_no_prefetch['Operation'] = pd.Categorical(combined_u_no_prefetch['Operation'], categories=classic_order, ordered=True)

# Group by Operation and compute mean and std
mean_df_u_no_prefetch = combined_u_no_prefetch.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_u_no_prefetch = combined_u_no_prefetch.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

# Save results to CSV
mean_df_u_no_prefetch.to_csv("Results_MI300X/unified_mean_no_prefetch.csv", index=False)
std_df_u_no_prefetch.to_csv("Results_MI300X/unified_std_no_prefetch.csv", index=False)


# === Load all CSVs (aligned with plotting cell expectations) ===
classic_mean = pd.read_csv("Results_MI300X/classic_mean.csv")
classic_std  = pd.read_csv("Results_MI300X/classic_std.csv")

# Unified (Prefetch)
unified_mean = pd.read_csv("Results_MI300X/unified_mean.csv")
unified_std  = pd.read_csv("Results_MI300X/unified_std.csv")

# Unified (No Prefetch)
unified_no_prefetch_mean = pd.read_csv("Results_MI300X/unified_mean_no_prefetch.csv")
unified_no_prefetch_std  = pd.read_csv("Results_MI300X/unified_std_no_prefetch.csv")


# Merges
merged_mean = classic_mean.merge(unified_mean, on='Operation', suffixes=('_Classic', '_Unified'))
merged_mean = merged_mean.merge(
    unified_no_prefetch_mean[['Operation', 'Time']].rename(columns={'Time': 'Time_NoPrefetch'}),
    on='Operation'
)

merged_std = classic_std.merge(unified_std, on='Operation', suffixes=('_Classic', '_Unified'))
merged_std = merged_std.merge(
    unified_no_prefetch_std[['Operation', 'Time']].rename(columns={'Time': 'Time_NoPrefetch'}),
    on='Operation'
)

# Save merged mean for the base bar chart 
merged_mean.to_csv("mean_timing_comparison.csv", index=False)









# 1.approval linear ===

# --- constants (as you had) ---
FS_BASE   = plt.rcParams.get('font.size', 8)
FS_TOP    = FS_BASE
TOP_DY    = 1
EPS       = 1e-6

def fmt_num(v):
    """Show no decimals if value > 10, else 2 decimals."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    return f"{int(round(v))}" if v > 10 else f"{v:.2f}"


# Column for the no-prefetch series
COL_U_NP = "Time_NoPrefetch"

def y_at_std_top(mean, std):
    if np.isfinite(mean) and np.isfinite(std):
        return max(mean + std, mean + EPS)
    return mean

# Index once
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

# Order on x-axis
SIMPLE_OPS = ["readDataFromFile", "allocateInputData", "allocateOutputData",
              "writeDataToFile", "freeInputData", "freeOutputData"]

groups = []
for op in SIMPLE_OPS[:3]: groups.append(("simple", op, op))
groups.append(("kernel", "Kernel", "Kernel"))   # single stacked "Kernel" group
for op in SIMPLE_OPS[3:]: groups.append(("simple", op, op))

x = np.arange(len(groups))
bar_width = 0.22
OFFSETS = (-bar_width, 0.0, +bar_width)  # Classic, Unified, NoPrefetch

plt.figure(figsize=(14, 8), constrained_layout=True)

# Colors
classic_exec = "#5790fc"; classic_sub  = "#2b477d"
unified_exec = "#f89c20"; unified_sub  = "#7a4d10"
nopref_exec  = "#e42536"; nopref_sub   = "#780f1c"

# Whisker colors
c_err = "#2f5597"; u_err = "#a15c00"; n_err = "#b31d2b"

# Legend labels (dedupe)
lbl_c_exe = "Classic-execution";        lbl_c_sub = "Classic-submission"
lbl_u_exe = "Unified-execution";        lbl_u_sub = "Unified-submission"
lbl_n_exe = "NoPrefetch-execution";     lbl_n_sub = "NoPrefetch-submission"
used_labels = set()
def add_bar(*args, label=None, **kw):
    if label and label in used_labels: label="_nolegend_"
    elif label: used_labels.add(label)
    return plt.bar(*args, label=label, **kw)

def add_errbar(xc, y, s, color):
    if np.isfinite(y) and np.isfinite(s) and y > EPS and s > 0:
        low = min(s, y - EPS)   # keep y-low > 0 if you use log elsewhere
        yerr = np.array([[low], [s]])
        plt.errorbar(xc, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.0, capsize=4, zorder=3)

# --- draw ---
for i, (kind, key, pretty) in enumerate(groups):
    xc = x[i] + OFFSETS[0]  # classic
    xu = x[i] + OFFSETS[1]  # unified
    xn = x[i] + OFFSETS[2]  # no prefetch

    if kind == "simple":
        c_val = get_mean(key, "Time_Classic")
        u_val = get_mean(key, "Time_Unified")
        n_val = get_mean(key, COL_U_NP)

        c_std = get_std(key, "Time_Classic")
        u_std = get_std(key, "Time_Unified")
        n_std = get_std(key, COL_U_NP)

        add_bar(xc, [c_val], width=bar_width, color=classic_exec, label=lbl_c_exe)
        add_bar(xu, [u_val], width=bar_width, color=unified_exec, label=lbl_u_exe)
        add_bar(xn, [n_val], width=bar_width, color=nopref_exec, label=lbl_n_exe)

        # labels at whisker tip (mean only)
        if np.isfinite(c_val):
            plt.annotate(fmt_num(c_val), (xc, y_at_std_top(c_val, c_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(u_val):
            plt.annotate(fmt_num(u_val), (xu, y_at_std_top(u_val, u_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(n_val):
            plt.annotate(fmt_num(n_val), (xn, y_at_std_top(n_val, n_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)


        add_errbar(xc, c_val, c_std, c_err)
        add_errbar(xu, u_val, u_std, u_err)
        add_errbar(xn, n_val, n_std, n_err)

    else:  # "Kernel" stacked using kernelSubmission/kernelExecution
        # Classic
        c_sub = get_mean("kernelSubmission", "Time_Classic")
        c_exe = get_mean("kernelExecution", "Time_Classic")
        c_top = max(c_exe - c_sub, 0.0) if np.isfinite(c_exe) and np.isfinite(c_sub) else np.nan
        c_exe_std = get_std("kernelExecution", "Time_Classic")

        # Unified
        u_sub = get_mean("kernelSubmission", "Time_Unified")
        u_exe = get_mean("kernelExecution", "Time_Unified")
        u_top = max(u_exe - u_sub, 0.0) if np.isfinite(u_exe) and np.isfinite(u_sub) else np.nan
        u_exe_std = get_std("kernelExecution", "Time_Unified")

        # No Prefetch
        n_sub = get_mean("kernelSubmission", COL_U_NP)
        n_exe = get_mean("kernelExecution", COL_U_NP)
        n_top = max(n_exe - n_sub, 0.0) if np.isfinite(n_exe) and np.isfinite(n_sub) else np.nan
        n_exe_std = get_std("kernelExecution", COL_U_NP)

        # Classic stack
        add_bar(xc, [c_sub], width=bar_width, color=classic_sub,  label=lbl_c_sub)
        add_bar(xc, [c_top], width=bar_width, bottom=[c_sub], color=classic_exec, label=lbl_c_exe)
        # Unified stack
        add_bar(xu, [u_sub], width=bar_width, color=unified_sub,  label=lbl_u_sub)
        add_bar(xu, [u_top], width=bar_width, bottom=[u_sub], color=unified_exec, label=lbl_u_exe)
        # No Prefetch stack
        add_bar(xn, [n_sub], width=bar_width, color=nopref_sub,   label=lbl_n_sub)
        add_bar(xn, [n_top], width=bar_width, bottom=[n_sub], color=nopref_exec, label=lbl_n_exe)

        # error bars (on total = execution)
        add_errbar(xc, c_exe, c_exe_std, c_err)
        add_errbar(xu, u_exe, u_exe_std, u_err)
        add_errbar(xn, n_exe, n_exe_std, n_err)

        # mean labels at whisker tip
        if np.isfinite(c_exe):
            plt.annotate(fmt_num(c_exe), (xc, y_at_std_top(c_exe, c_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(u_exe):
            plt.annotate(fmt_num(u_exe), (xu, y_at_std_top(u_exe, u_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(n_exe):
            plt.annotate(fmt_num(n_exe), (xn, y_at_std_top(n_exe, n_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

# Axes/legend
plt.xticks(x, [g[2] for g in groups], rotation=15, ha="right", rotation_mode="anchor")
plt.ylabel("Time (ms)")
plt.title("Classic vs Unified (Prefetch / No Prefetch)")
plt.grid(axis="y", linestyle="--", alpha=0.6)

from matplotlib.lines import Line2D
err_proxy = Line2D([0],[0], color="#444", lw=1, label="±1σ (std)")
handles, labels = plt.gca().get_legend_handles_labels()
if "±1σ (std)" not in labels: handles.append(err_proxy); labels.append("±1σ (std)")
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

plt.savefig("Results_MI300X/approval_linear.png", dpi=300, bbox_inches="tight")
plt.show()








# 1.2 approval log ===

# --- constants (as you had) ---
FS_BASE   = plt.rcParams.get('font.size', 8)
FS_TOP    = FS_BASE
TOP_DY    = 1
EPS       = 1e-6

def fmt_num(v):
    """Show no decimals if value > 10, else 2 decimals."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    return f"{int(round(v))}" if v > 10 else f"{v:.2f}"


# Column for the no-prefetch series
COL_U_NP = "Time_NoPrefetch"

def y_at_std_top(mean, std):
    if np.isfinite(mean) and np.isfinite(std):
        return max(mean + std, mean + EPS)
    return mean

# Index once
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

# Order on x-axis
SIMPLE_OPS = ["readDataFromFile", "allocateInputData", "allocateOutputData",
              "writeDataToFile", "freeInputData", "freeOutputData"]

groups = []
for op in SIMPLE_OPS[:3]: groups.append(("simple", op, op))
groups.append(("kernel", "Kernel", "Kernel"))   # single stacked "Kernel" group
for op in SIMPLE_OPS[3:]: groups.append(("simple", op, op))

x = np.arange(len(groups))
bar_width = 0.22
OFFSETS = (-bar_width, 0.0, +bar_width)  # Classic, Unified, NoPrefetch

plt.figure(figsize=(14, 8), constrained_layout=True)

# Colors
classic_exec = "#5790fc"; classic_sub  = "#2b477d"
unified_exec = "#f89c20"; unified_sub  = "#7a4d10"
nopref_exec  = "#e42536"; nopref_sub   = "#780f1c"

# Whisker colors
c_err = "#2f5597"; u_err = "#a15c00"; n_err = "#b31d2b"

# Legend labels (dedupe)
lbl_c_exe = "Classic-execution";        lbl_c_sub = "Classic-submission"
lbl_u_exe = "Unified-execution";        lbl_u_sub = "Unified-submission"
lbl_n_exe = "NoPrefetch-execution";     lbl_n_sub = "NoPrefetch-submission"
used_labels = set()
def add_bar(*args, label=None, **kw):
    if label and label in used_labels: label="_nolegend_"
    elif label: used_labels.add(label)
    return plt.bar(*args, label=label, **kw)

def add_errbar(xc, y, s, color):
    if np.isfinite(y) and np.isfinite(s) and y > EPS and s > 0:
        low = min(s, y - EPS)   # keep y-low > 0 if you use log elsewhere
        yerr = np.array([[low], [s]])
        plt.errorbar(xc, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.0, capsize=4, zorder=3)

# --- draw ---
for i, (kind, key, pretty) in enumerate(groups):
    xc = x[i] + OFFSETS[0]  # classic
    xu = x[i] + OFFSETS[1]  # unified
    xn = x[i] + OFFSETS[2]  # no prefetch

    if kind == "simple":
        c_val = get_mean(key, "Time_Classic")
        u_val = get_mean(key, "Time_Unified")
        n_val = get_mean(key, COL_U_NP)

        c_std = get_std(key, "Time_Classic")
        u_std = get_std(key, "Time_Unified")
        n_std = get_std(key, COL_U_NP)

        add_bar(xc, [c_val], width=bar_width, color=classic_exec, label=lbl_c_exe)
        add_bar(xu, [u_val], width=bar_width, color=unified_exec, label=lbl_u_exe)
        add_bar(xn, [n_val], width=bar_width, color=nopref_exec, label=lbl_n_exe)

        # labels at whisker tip (mean only)
        if np.isfinite(c_val):
            plt.annotate(fmt_num(c_val), (xc, y_at_std_top(c_val, c_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(u_val):
            plt.annotate(fmt_num(u_val), (xu, y_at_std_top(u_val, u_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(n_val):
            plt.annotate(fmt_num(n_val), (xn, y_at_std_top(n_val, n_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)


        add_errbar(xc, c_val, c_std, c_err)
        add_errbar(xu, u_val, u_std, u_err)
        add_errbar(xn, n_val, n_std, n_err)

    else:  # "Kernel" stacked using kernelSubmission/kernelExecution
        # Classic
        c_sub = get_mean("kernelSubmission", "Time_Classic")
        c_exe = get_mean("kernelExecution", "Time_Classic")
        c_top = max(c_exe - c_sub, 0.0) if np.isfinite(c_exe) and np.isfinite(c_sub) else np.nan
        c_exe_std = get_std("kernelExecution", "Time_Classic")

        # Unified
        u_sub = get_mean("kernelSubmission", "Time_Unified")
        u_exe = get_mean("kernelExecution", "Time_Unified")
        u_top = max(u_exe - u_sub, 0.0) if np.isfinite(u_exe) and np.isfinite(u_sub) else np.nan
        u_exe_std = get_std("kernelExecution", "Time_Unified")

        # No Prefetch
        n_sub = get_mean("kernelSubmission", COL_U_NP)
        n_exe = get_mean("kernelExecution", COL_U_NP)
        n_top = max(n_exe - n_sub, 0.0) if np.isfinite(n_exe) and np.isfinite(n_sub) else np.nan
        n_exe_std = get_std("kernelExecution", COL_U_NP)

        # Classic stack
        add_bar(xc, [c_sub], width=bar_width, color=classic_sub,  label=lbl_c_sub)
        add_bar(xc, [c_top], width=bar_width, bottom=[c_sub], color=classic_exec, label=lbl_c_exe)
        # Unified stack
        add_bar(xu, [u_sub], width=bar_width, color=unified_sub,  label=lbl_u_sub)
        add_bar(xu, [u_top], width=bar_width, bottom=[u_sub], color=unified_exec, label=lbl_u_exe)
        # No Prefetch stack
        add_bar(xn, [n_sub], width=bar_width, color=nopref_sub,   label=lbl_n_sub)
        add_bar(xn, [n_top], width=bar_width, bottom=[n_sub], color=nopref_exec, label=lbl_n_exe)

        # error bars (on total = execution)
        add_errbar(xc, c_exe, c_exe_std, c_err)
        add_errbar(xu, u_exe, u_exe_std, u_err)
        add_errbar(xn, n_exe, n_exe_std, n_err)

        # mean labels at whisker tip
        if np.isfinite(c_exe):
            plt.annotate(fmt_num(c_exe), (xc, y_at_std_top(c_exe, c_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(u_exe):
            plt.annotate(fmt_num(u_exe), (xu, y_at_std_top(u_exe, u_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(n_exe):
            plt.annotate(fmt_num(n_exe), (xn, y_at_std_top(n_exe, n_exe_std)), xytext=(0, TOP_DY),
                        textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)


# Axes/legend
plt.xticks(x, [g[2] for g in groups], rotation=15, ha="right", rotation_mode="anchor")
plt.ylabel("Time (ms)")
plt.yscale('log')  # logarithmic scale
plt.title("Classic vs Unified (Prefetch / No Prefetch) — Log scale")
plt.grid(axis="y", linestyle="--", alpha=0.6)

from matplotlib.lines import Line2D
err_proxy = Line2D([0],[0], color="#444", lw=1, label="±1σ (std)")
handles, labels = plt.gca().get_legend_handles_labels()
if "±1σ (std)" not in labels: handles.append(err_proxy); labels.append("±1σ (std)")
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

plt.savefig("Results_MI300X/approval_log.png", dpi=300, bbox_inches="tight")
plt.show()





# 1.3 approval linear zoom in ===

# --- constants (as you had) ---
FS_BASE   = plt.rcParams.get('font.size', 8)
FS_TOP    = FS_BASE
TOP_DY    = 1
EPS       = 1e-6

# Column for the no-prefetch series
COL_U_NP = "Time_NoPrefetch"

def y_at_std_top(mean, std):
    if np.isfinite(mean) and np.isfinite(std):
        return max(mean + std, mean + EPS)
    return mean

# Index once
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

# Order on x-axis
SIMPLE_OPS = ["readDataFromFile", "allocateInputData", "allocateOutputData",
              "writeDataToFile", "freeInputData", "freeOutputData"]

groups = []
for op in SIMPLE_OPS[:3]: groups.append(("simple", op, op))
groups.append(("kernel", "Kernel", "Kernel"))   # single stacked "Kernel" group
for op in SIMPLE_OPS[3:]: groups.append(("simple", op, op))

x = np.arange(len(groups))
bar_width = 0.22
OFFSETS = (-bar_width, 0.0, +bar_width)  # Classic, Unified, NoPrefetch

plt.figure(figsize=(14, 8), constrained_layout=True)

# Colors
classic_exec = "#5790fc"; classic_sub  = "#2b477d"
unified_exec = "#f89c20"; unified_sub  = "#7a4d10"
nopref_exec  = "#e42536"; nopref_sub   = "#780f1c"

# Whisker colors
c_err = "#2f5597"; u_err = "#a15c00"; n_err = "#b31d2b"

# Legend labels (dedupe)
lbl_c_exe = "Classic-execution";        lbl_c_sub = "Classic-submission"
lbl_u_exe = "Unified-execution";        lbl_u_sub = "Unified-submission"
lbl_n_exe = "NoPrefetch-execution";     lbl_n_sub = "NoPrefetch-submission"
used_labels = set()
def add_bar(*args, label=None, **kw):
    if label and label in used_labels: label="_nolegend_"
    elif label: used_labels.add(label)
    return plt.bar(*args, label=label, **kw)

def add_errbar(xc, y, s, color):
    if np.isfinite(y) and np.isfinite(s) and y > EPS and s > 0:
        low = min(s, y - EPS)   # keep y-low > 0 if you use log elsewhere
        yerr = np.array([[low], [s]])
        plt.errorbar(xc, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.0, capsize=4, zorder=3)

# --- draw ---
for i, (kind, key, pretty) in enumerate(groups):
    xc = x[i] + OFFSETS[0]  # classic
    xu = x[i] + OFFSETS[1]  # unified
    xn = x[i] + OFFSETS[2]  # no prefetch

    if kind == "simple":
        c_val = get_mean(key, "Time_Classic")
        u_val = get_mean(key, "Time_Unified")
        n_val = get_mean(key, COL_U_NP)

        c_std = get_std(key, "Time_Classic")
        u_std = get_std(key, "Time_Unified")
        n_std = get_std(key, COL_U_NP)

        add_bar(xc, [c_val], width=bar_width, color=classic_exec, label=lbl_c_exe)
        add_bar(xu, [u_val], width=bar_width, color=unified_exec, label=lbl_u_exe)
        add_bar(xn, [n_val], width=bar_width, color=nopref_exec, label=lbl_n_exe)

        # labels at whisker tip (mean only)
        if np.isfinite(c_val): plt.annotate(f"{c_val:.2f}", (xc, y_at_std_top(c_val,c_std)), xytext=(0, TOP_DY),
                                            textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(u_val): plt.annotate(f"{u_val:.2f}", (xu, y_at_std_top(u_val,u_std)), xytext=(0, TOP_DY),
                                            textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(n_val): plt.annotate(f"{n_val:.2f}", (xn, y_at_std_top(n_val,n_std)), xytext=(0, TOP_DY),
                                            textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

        add_errbar(xc, c_val, c_std, c_err)
        add_errbar(xu, u_val, u_std, u_err)
        add_errbar(xn, n_val, n_std, n_err)

    else:  # "Kernel" stacked using kernelSubmission/kernelExecution
        # Classic
        c_sub = get_mean("kernelSubmission", "Time_Classic")
        c_exe = get_mean("kernelExecution", "Time_Classic")
        c_top = max(c_exe - c_sub, 0.0) if np.isfinite(c_exe) and np.isfinite(c_sub) else np.nan
        c_exe_std = get_std("kernelExecution", "Time_Classic")

        # Unified
        u_sub = get_mean("kernelSubmission", "Time_Unified")
        u_exe = get_mean("kernelExecution", "Time_Unified")
        u_top = max(u_exe - u_sub, 0.0) if np.isfinite(u_exe) and np.isfinite(u_sub) else np.nan
        u_exe_std = get_std("kernelExecution", "Time_Unified")

        # No Prefetch
        n_sub = get_mean("kernelSubmission", COL_U_NP)
        n_exe = get_mean("kernelExecution", COL_U_NP)
        n_top = max(n_exe - n_sub, 0.0) if np.isfinite(n_exe) and np.isfinite(n_sub) else np.nan
        n_exe_std = get_std("kernelExecution", COL_U_NP)

        # Classic stack
        add_bar(xc, [c_sub], width=bar_width, color=classic_sub,  label=lbl_c_sub)
        add_bar(xc, [c_top], width=bar_width, bottom=[c_sub], color=classic_exec, label=lbl_c_exe)
        # Unified stack
        add_bar(xu, [u_sub], width=bar_width, color=unified_sub,  label=lbl_u_sub)
        add_bar(xu, [u_top], width=bar_width, bottom=[u_sub], color=unified_exec, label=lbl_u_exe)
        # No Prefetch stack
        add_bar(xn, [n_sub], width=bar_width, color=nopref_sub,   label=lbl_n_sub)
        add_bar(xn, [n_top], width=bar_width, bottom=[n_sub], color=nopref_exec, label=lbl_n_exe)

        # error bars (on total = execution)
        add_errbar(xc, c_exe, c_exe_std, c_err)
        add_errbar(xu, u_exe, u_exe_std, u_err)
        add_errbar(xn, n_exe, n_exe_std, n_err)

        # mean labels at whisker tip
        if np.isfinite(c_exe): plt.annotate(f"{c_exe:.2f}", (xc, y_at_std_top(c_exe,c_exe_std)), xytext=(0, TOP_DY),
                                            textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(u_exe): plt.annotate(f"{u_exe:.2f}", (xu, y_at_std_top(u_exe,u_exe_std)), xytext=(0, TOP_DY),
                                            textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)
        if np.isfinite(n_exe): plt.annotate(f"{n_exe:.2f}", (xn, y_at_std_top(n_exe,n_exe_std)), xytext=(0, TOP_DY),
                                            textcoords="offset points", ha="center", va="bottom", fontsize=FS_TOP)

# Axes/legend
plt.xticks(x, [g[2] for g in groups], rotation=15, ha="right", rotation_mode="anchor")
plt.ylabel("Time (ms)")
plt.title("Classic vs Unified (Prefetch / No Prefetch) — with Kernel stacked")
plt.grid(axis="y", linestyle="--", alpha=0.6)

from matplotlib.lines import Line2D
err_proxy = Line2D([0],[0], color="#444", lw=1, label="±1σ (std)")
handles, labels = plt.gca().get_legend_handles_labels()
if "±1σ (std)" not in labels: handles.append(err_proxy); labels.append("±1σ (std)")
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

# --- Also save a zoomed-in version (linear 0–10 ms) ---
ax = plt.gca()
ax.set_yscale('linear')
ax.set_ylim(0, 8)                         # pick 4, 6, 8, 10… as you like
ax.set_autoscale_on(False) 
ax.set_title("Classic vs Unified (Prefetch / No Prefetch) — Zoomed 0–8 ms")

plt.savefig("Results_MI300X/approval_linear_zoom_in.png", dpi=300, bbox_inches="tight")

plt.show()