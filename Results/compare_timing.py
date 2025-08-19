import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
import glob
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Collect all classicX.csv files
classic_files = sorted(glob.glob("../gitlab_clue/Results/results_classic*.csv"))
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
mean_df_c.to_csv("../gitlab_clue/Results/classic_mean.csv", index=False)
std_df_c.to_csv("../gitlab_clue/Results/classic_std.csv", index=False)


# Collect all classicX.csv files
unified_files = sorted(glob.glob("Results/results_unified*.csv"))
unified_dfs = [pd.read_csv(f) for f in unified_files if not f.endswith("results_unified0.csv")]

# Combine all runs into one DataFrame
combined_u = pd.concat(unified_dfs)

# Use Categorical to enforce the original order
combined_u['Operation'] = pd.Categorical(combined_u['Operation'], categories=classic_order, ordered=True)

# Group by Operation and compute mean and std
mean_df_u = combined_u.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_u = combined_u.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

# Save results to CSV
mean_df_u.to_csv("Results/unified_mean.csv", index=False)
std_df_u.to_csv("Results/unified_std.csv", index=False)






# === Load all CSVs ===
classic_mean = pd.read_csv("../gitlab_clue/Results/classic_mean.csv")
unified_mean = pd.read_csv("Results/unified_mean.csv")
classic_std = pd.read_csv("../gitlab_clue/Results/classic_std.csv")
unified_std = pd.read_csv("Results/unified_std.csv")

# === Merge mean and std separately ===
merged_mean = classic_mean.merge(unified_mean, on='Operation', suffixes=('_Classic', '_Unified'))
merged_std = classic_std.merge(unified_std, on='Operation', suffixes=('_Classic', '_Unified'))

# Save merged mean for the base bar chart
merged_mean.to_csv("mean_timing_comparison.csv", index=False)


# 1.approval log ===

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
classic_exec = "#5790fc"; classic_sub  = "#964a8b"
unified_exec = "#f89c20"; unified_sub  = "#e42536"

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
plt.yscale('log')  # logarithmic scale
plt.title("Classic vs Unified Memory - Log scale")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Add a proxy legend entry for error bars (avoid duplicate whiskers)
from matplotlib.lines import Line2D
err_proxy = Line2D([0], [0], color="#444", lw=1, label="±1σ (std)")
handles, labels = plt.gca().get_legend_handles_labels()
if "±1σ (std)" not in labels:
    handles.append(err_proxy); labels.append("±1σ (std)")
plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

plt.savefig("Results/approval_log.png", dpi=300, bbox_inches="tight")
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
classic_exec = "#5790fc"; classic_sub  = "#964a8b"
unified_exec = "#f89c20"; unified_sub  = "#e42536"

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

plt.savefig("Results/approval_linear.png", dpi=300, bbox_inches="tight")
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
classic_exec = "#5790fc"; classic_sub  = "#964a8b"
unified_exec = "#f89c20"; unified_sub  = "#e42536"

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
ax.set_ylim(0, 4)                         # pick 4, 6, 8, 10… as you like
ax.set_autoscale_on(False) 
ax.set_title("Classic vs Unified Memory — Zoomed 0–4 ms")

plt.savefig("Results/approval_linear_zoom_in.png", dpi=300, bbox_inches="tight")

plt.show()