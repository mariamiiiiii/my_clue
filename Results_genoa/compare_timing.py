import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Collect all results_classicX.csv files
classic_files = sorted(glob.glob("../gitlab_clue/Results_genoa/results_classic*.csv"))
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
mean_df_c.to_csv("../gitlab_clue/Results_genoa/classic_mean.csv", index=False)
std_df_c.to_csv("../gitlab_clue/Results_genoa/classic_std.csv", index=False)


# Collect all results_unifiedX.csv files
unified_files = sorted(glob.glob("Results_genoa/results_unified*.csv"))
unified_dfs = [pd.read_csv(f) for f in unified_files if not f.endswith("results_unified0.csv")]

# Combine all runs into one DataFrame
combined_u = pd.concat(unified_dfs)

# Use Categorical to enforce the original order
combined_u['Operation'] = pd.Categorical(combined_u['Operation'], categories=classic_order, ordered=True)

# Group by Operation and compute mean and std
mean_df_u = combined_u.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_u = combined_u.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

# Save results to CSV
mean_df_u.to_csv("Results_genoa/unified_mean.csv", index=False)
std_df_u.to_csv("Results_genoa/unified_std.csv", index=False)


# Collect all results_unified_no_prefetchX.csv files
unified_no_prefetch_files = sorted(glob.glob("Results_genoa/results_unified_no_prefetch*.csv"))
unified_no_prefetch_dfs = [pd.read_csv(f) for f in unified_no_prefetch_files if not f.endswith("results_unified_no_prefetch0.csv")]

# Combine all runs into one DataFrame
combined_u_no_prefetch = pd.concat(unified_no_prefetch_dfs)

# Use Categorical to enforce the original order
combined_u_no_prefetch['Operation'] = pd.Categorical(combined_u_no_prefetch['Operation'], categories=classic_order, ordered=True)

# Group by Operation and compute mean and std
mean_df_u_no_prefetch = combined_u_no_prefetch.groupby("Operation", sort=False, observed=False)["Time"].mean().reset_index()
std_df_u_no_prefetch = combined_u_no_prefetch.groupby("Operation", sort=False, observed=False)["Time"].std().reset_index()

# Save results to CSV
mean_df_u_no_prefetch.to_csv("Results_genoa/unified_mean_no_prefetch.csv", index=False)
std_df_u_no_prefetch.to_csv("Results_genoa/unified_std_no_prefetch.csv", index=False)



# === Load all CSVs (aligned with plotting cell expectations) ===
classic_mean = pd.read_csv("../gitlab_clue/Results_genoa/classic_mean.csv")
classic_std  = pd.read_csv("../gitlab_clue/Results_genoa/classic_std.csv")

# Unified (Prefetch)
unified_mean = pd.read_csv("Results_genoa/unified_mean.csv")
unified_std  = pd.read_csv("Results_genoa/unified_std.csv")

# Unified (No Prefetch)
unified_no_prefetch_mean = pd.read_csv("Results_genoa/unified_mean_no_prefetch.csv")
unified_no_prefetch_std  = pd.read_csv("Results_genoa/unified_std_no_prefetch.csv")


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

# === Common setup ===
df = merged_mean
plt.rcParams.update({
    'font.size': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'axes.titlesize': 10
})

bar_width = 0.27
index = np.arange(len(df))  # use numpy arange for convenience

# x positions for the three bars per Operation
x_copy       = index
x_prefetch   = index + bar_width
x_noprefetch = index + 2 * bar_width

# Helper to add value labels
def _add_labels(bars, offset):
    for bar in bars:
        yval = bar.get_height()
        label = f'{yval:.3f}'.rstrip('0').rstrip('.')
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, label,
                 ha='center', va='bottom', fontsize=8)

# === Plot 1.1: Basic mean comparison (no error bars) ===
plt.figure(figsize=(14, 8))
bars_copy = plt.bar(x_copy, df['Time_Classic'], bar_width, label='Copy', color='#4682B4')
bars_prefetch = plt.bar(x_prefetch, df['Time_Unified'], bar_width, label='Prefetch', color='#CD5C5C')
bars_noprefetch = plt.bar(x_noprefetch, df['Time_NoPrefetch'], bar_width, label='No Prefetch', color='#2E8B57')

_add_labels(bars_copy, 0.5)
_add_labels(bars_prefetch, 0.5)
_add_labels(bars_noprefetch, 0.5)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Performance Comparison: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results_genoa/mean_comparison.png")
plt.show()

# === Plot 1.2: Basic mean comparison (zoom in, 0–10 ms) ===
plt.figure(figsize=(14, 8))
bars_copy = plt.bar(x_copy, df['Time_Classic'], bar_width, label='Copy', color='#4682B4')
bars_prefetch = plt.bar(x_prefetch, df['Time_Unified'], bar_width, label='Prefetch', color='#CD5C5C')
bars_noprefetch = plt.bar(x_noprefetch, df['Time_NoPrefetch'], bar_width, label='No Prefetch', color='#2E8B57')

_add_labels(bars_copy, 0.04)
_add_labels(bars_prefetch, 0.04)
_add_labels(bars_noprefetch, 0.04)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Performance Comparison: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 10)
plt.savefig("Results_genoa/mean_comparison_zoom_in.png")
plt.show()

# === Plot 1.3: Basic mean comparison (log scale) ===
plt.figure(figsize=(14, 8))
bars_copy = plt.bar(x_copy, df['Time_Classic'], bar_width, label='Copy', color='#4682B4')
bars_prefetch = plt.bar(x_prefetch, df['Time_Unified'], bar_width, label='Prefetch', color='#CD5C5C')
bars_noprefetch = plt.bar(x_noprefetch, df['Time_NoPrefetch'], bar_width, label='No Prefetch', color='#2E8B57')

_add_labels(bars_copy, 0.5)
_add_labels(bars_prefetch, 0.5)
_add_labels(bars_noprefetch, 0.5)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Performance Comparison: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results_genoa/mean_comparison_log.png")
plt.show()

# === Plot 2.1: Mean + Std Error Bars ===
plt.figure(figsize=(14, 8))

bars_copy = plt.bar(
    x_copy, df['Time_Classic'],
    yerr=merged_std['Time_Classic'], capsize=5,
    label='Copy', width=bar_width, color='#4682B4', alpha=0.8
)
bars_prefetch = plt.bar(
    x_prefetch, df['Time_Unified'],
    yerr=merged_std['Time_Unified'], capsize=5,
    label='Prefetch', width=bar_width, color='#CD5C5C', alpha=0.8
)
bars_noprefetch = plt.bar(
    x_noprefetch, df['Time_NoPrefetch'],
    yerr=merged_std['Time_NoPrefetch'], capsize=5,
    label='No Prefetch', width=bar_width, color='#2E8B57', alpha=0.8
)

_add_labels(bars_copy, 0.5)
_add_labels(bars_prefetch, 0.5)
_add_labels(bars_noprefetch, 0.5)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Mean ± Std Deviation: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results_genoa/mean_std_copy_prefetch_noprefetch.png")
plt.show()

# === Plot 2.2: Mean + Std Error Bars (zoom in) ===
plt.figure(figsize=(14, 8))

bars_copy = plt.bar(
    x_copy, df['Time_Classic'],
    yerr=merged_std['Time_Classic'], capsize=5,
    label='Copy', width=bar_width, color='#4682B4', alpha=0.8
)
bars_prefetch = plt.bar(
    x_prefetch, df['Time_Unified'],
    yerr=merged_std['Time_Unified'], capsize=5,
    label='Prefetch', width=bar_width, color='#CD5C5C', alpha=0.8
)
bars_noprefetch = plt.bar(
    x_noprefetch, df['Time_NoPrefetch'],
    yerr=merged_std['Time_NoPrefetch'], capsize=5,
    label='No Prefetch', width=bar_width, color='#2E8B57', alpha=0.8
)

_add_labels(bars_copy, 0.04)
_add_labels(bars_prefetch, 0.04)
_add_labels(bars_noprefetch, 0.04)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Mean ± Std Deviation: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 10)
plt.savefig("Results_genoa/mean_std_copy_prefetch_noprefetch_zoom_in.png")
plt.show()

# === Plot 2.3: Mean + Std Error Bars (log) ===
plt.figure(figsize=(14, 8))

bars_copy = plt.bar(
    x_copy, df['Time_Classic'],
    yerr=merged_std['Time_Classic'], capsize=5,
    label='Copy', width=bar_width, color='#4B7D74', ecolor='#B46841', alpha=0.8
)
bars_prefetch = plt.bar(
    x_prefetch, df['Time_Unified'],
    yerr=merged_std['Time_Unified'], capsize=5,
    label='Prefetch', width=bar_width, color='#D9C89E', ecolor='#B46841', alpha=0.8
)
bars_noprefetch = plt.bar(
    x_noprefetch, df['Time_NoPrefetch'],
    yerr=merged_std['Time_NoPrefetch'], capsize=5,
    label='No Prefetch', width=bar_width, color='#8FBC8F', ecolor='#B46841', alpha=0.8
)

# Position labels a bit above bar+error for log plot
def _label_above_error(bars, std_series):
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        std = std_series.iloc[i]
        label = f'{yval:.3f}'.rstrip('0').rstrip('.')
        ypos = yval * (1.05 + (std / yval if yval != 0 else 0))
        plt.text(bar.get_x() + bar.get_width()/2, ypos, label,
                 ha='center', va='bottom', fontsize=8)

_label_above_error(bars_copy, merged_std['Time_Classic'])
_label_above_error(bars_prefetch, merged_std['Time_Unified'])
_label_above_error(bars_noprefetch, merged_std['Time_NoPrefetch'])

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Mean ± Std Deviation: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results_genoa/mean_std_copy_prefetch_noprefetch_log.png")
plt.show()

# === Plot 3.1: Std Deviation Comparison Only ===
plt.figure(figsize=(14, 8))

bars_copy_std = plt.bar(x_copy, merged_std['Time_Classic'], width=bar_width, label='Copy STD', color='#1E90FF')
bars_prefetch_std = plt.bar(x_prefetch, merged_std['Time_Unified'], width=bar_width, label='Prefetch STD', color='#FF7F7F')
bars_noprefetch_std = plt.bar(x_noprefetch, merged_std['Time_NoPrefetch'], width=bar_width, label='No Prefetch STD', color='#66CDAA')

_add_labels(bars_copy_std, 0.5)
_add_labels(bars_prefetch_std, 0.5)
_add_labels(bars_noprefetch_std, 0.5)

plt.xlabel('Operation')
plt.ylabel('Standard Deviation (ms)')
plt.title('Standard Deviation Comparison: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results_genoa/std_comparison.png")
plt.show()

# === Plot 3.2: Std Deviation Comparison Only (zoom in) ===
plt.figure(figsize=(14, 8))

bars_copy_std = plt.bar(x_copy, merged_std['Time_Classic'], width=bar_width, label='Copy STD', color='#1E90FF')
bars_prefetch_std = plt.bar(x_prefetch, merged_std['Time_Unified'], width=bar_width, label='Prefetch STD', color='#FF7F7F')
bars_noprefetch_std = plt.bar(x_noprefetch, merged_std['Time_NoPrefetch'], width=bar_width, label='No Prefetch STD', color='#66CDAA')

_add_labels(bars_copy_std, 0.01)
_add_labels(bars_prefetch_std, 0.01)
_add_labels(bars_noprefetch_std, 0.01)

plt.xlabel('Operation')
plt.ylabel('Standard Deviation (ms)')
plt.title('Standard Deviation Comparison: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 4)
plt.savefig("Results_genoa/std_comparison_zoom_in.png")
plt.show()

# === Plot 3.3: Std Deviation Comparison Only (log) ===
plt.figure(figsize=(14, 8))

bars_copy_std = plt.bar(x_copy, merged_std['Time_Classic'], width=bar_width, label='Copy STD', color='#1E90FF')
bars_prefetch_std = plt.bar(x_prefetch, merged_std['Time_Unified'], width=bar_width, label='Prefetch STD', color='#FF7F7F')
bars_noprefetch_std = plt.bar(x_noprefetch, merged_std['Time_NoPrefetch'], width=bar_width, label='No Prefetch STD', color='#66CDAA')

_add_labels(bars_copy_std, 0.5)
_add_labels(bars_prefetch_std, 0.5)
_add_labels(bars_noprefetch_std, 0.5)

plt.xlabel('Operation')
plt.ylabel('Standard Deviation (ms)')
plt.title('Standard Deviation Comparison: Copy vs Prefetch vs No Prefetch')
plt.xticks(index + bar_width, df['Operation'], rotation=15)
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results_genoa/std_comparison_log.png")
plt.show()