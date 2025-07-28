import pandas as pd
import matplotlib.pyplot as plt
import warnings
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

# === Plot 1.1: Basic mean comparison (no error bars) ===
df = merged_mean
plt.rcParams.update({
    'font.size': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'axes.titlesize': 10
})

bar_width = 0.35
index = range(len(df))

plt.figure(figsize=(14, 8))
bars_classic = plt.bar(index, df['Time_Classic'], bar_width, label='Classic', color='#4682B4')
bars_unified = plt.bar([i + bar_width for i in index], df['Time_Unified'], bar_width, label='Unified Memory', color='#CD5C5C')

# Add value labels
for bar in bars_classic + bars_unified:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, label, ha='center', va='bottom', fontsize=8)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Performance Comparison: Classic vs Unified Memory')
plt.xticks([i + bar_width / 2 for i in index], df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results/mean_comparison.png")
plt.show()

# === Plot 1.2: Basic mean comparison (no error bars) zoom in ===

plt.figure(figsize=(14, 8))
bars_classic = plt.bar(index, df['Time_Classic'], bar_width, label='Classic', color='#4682B4')
bars_unified = plt.bar([i + bar_width for i in index], df['Time_Unified'], bar_width, label='Unified Memory', color='#CD5C5C')

# Add value labels
for bar in bars_classic + bars_unified:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, label, ha='center', va='bottom', fontsize=8)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Performance Comparison: Classic vs Unified Memory')
plt.xticks([i + bar_width / 2 for i in index], df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 4)
plt.savefig("Results/mean_comparison_zoom_in.png")
plt.show()


# === Plot 2: Mean + Std Error Bars ===

plt.figure(figsize=(14, 8))

bars_classic_std = plt.bar(index, df['Time_Classic'], yerr=merged_std['Time_Classic'], capsize=5,
                           label='Classic', width=bar_width, color='#4682B4', alpha=0.8)
bars_unified_std = plt.bar([i + bar_width for i in index], df['Time_Unified'], yerr=merged_std['Time_Unified'], capsize=5,
                           label='Unified Memory', width=bar_width, color='#CD5C5C', alpha=0.8)

# Add value labels
for bar in bars_classic_std + bars_unified_std:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, label, ha='center', va='bottom', fontsize=8)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Mean ± Std Deviation: Classic vs Unified Memory')
plt.xticks([i + bar_width / 2 for i in index], df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results/mean_std_classic_vs_unified.png")
plt.show()
plt.ylim(0, 5)
plt.savefig("Results/mean_std_classic_vs_unified_zoom_in.png")
plt.show()

# === Plot 2.1: Mean + Std Error Bars ===

plt.figure(figsize=(14, 8))

bars_classic_std = plt.bar(index, df['Time_Classic'], yerr=merged_std['Time_Classic'], capsize=5,
                           label='Classic', width=bar_width, color='#4682B4', alpha=0.8)
bars_unified_std = plt.bar([i + bar_width for i in index], df['Time_Unified'], yerr=merged_std['Time_Unified'], capsize=5,
                           label='Unified Memory', width=bar_width, color='#CD5C5C', alpha=0.8)

# Add value labels
for bar in bars_classic_std + bars_unified_std:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, label, ha='center', va='bottom', fontsize=8)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Mean ± Std Deviation: Classic vs Unified Memory')
plt.xticks([i + bar_width / 2 for i in index], df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results/mean_std_classic_vs_unified.png")
plt.show()

# === Plot 2.1: Mean + Std Error Bars zoom in ===

plt.figure(figsize=(14, 8))

bars_classic_std = plt.bar(index, df['Time_Classic'], yerr=merged_std['Time_Classic'], capsize=5,
                           label='Classic', width=bar_width, color='#4682B4', alpha=0.8)
bars_unified_std = plt.bar([i + bar_width for i in index], df['Time_Unified'], yerr=merged_std['Time_Unified'], capsize=5,
                           label='Unified Memory', width=bar_width, color='#CD5C5C', alpha=0.8)

# Add value labels
for bar in bars_classic_std + bars_unified_std:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, label, ha='center', va='bottom', fontsize=8)

plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Mean ± Std Deviation: Classic vs Unified Memory')
plt.xticks([i + bar_width / 2 for i in index], df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 4)
plt.savefig("Results/mean_std_classic_vs_unified_zoom_in.png")
plt.show()

# === Plot 3.1: Std Deviation Comparison Only ===

plt.figure(figsize=(14, 8))

bars_classic_only_std = plt.bar(index, merged_std['Time_Classic'], width=bar_width, label='Classic STD', color='#1E90FF')
bars_unified_only_std = plt.bar([i + bar_width for i in index], merged_std['Time_Unified'], width=bar_width, label='Unified STD', color='#FF7F7F')

# Add value labels
for bar in bars_classic_only_std + bars_unified_only_std:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, label, ha='center', va='bottom', fontsize=8)

plt.xlabel('Operation')
plt.ylabel('Standard Deviation (ms)')
plt.title('Standard Deviation Comparison: Classic vs Unified Memory')
plt.xticks([i + bar_width / 2 for i in index], df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("Results/std_comparison.png")
plt.show()

# === Plot 3: Std Deviation Comparison Only ===

plt.figure(figsize=(14, 8))

bars_classic_only_std = plt.bar(index, merged_std['Time_Classic'], width=bar_width, label='Classic STD', color='#1E90FF')
bars_unified_only_std = plt.bar([i + bar_width for i in index], merged_std['Time_Unified'], width=bar_width, label='Unified STD', color='#FF7F7F')

# Add value labels
for bar in bars_classic_only_std + bars_unified_only_std:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, label, ha='center', va='bottom', fontsize=8)

plt.xlabel('Operation')
plt.ylabel('Standard Deviation (ms)')
plt.title('Standard Deviation Comparison: Classic vs Unified Memory')
plt.xticks([i + bar_width / 2 for i in index], df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1)
plt.savefig("Results/std_comparison_zoom_in.png")
plt.show()
