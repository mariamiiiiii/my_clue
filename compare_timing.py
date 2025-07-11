import pandas as pd
import matplotlib.pyplot as plt

# Load both CSVs
classic = pd.read_csv('../gitlab_clue/results_classic.csv')
unified = pd.read_csv('results_unified.csv')

# Merge on 'Operation'
merged = classic.merge(unified, on='Operation', suffixes=('_Classic', '_Unified'))

# Save merged file
merged.to_csv('timing_comparison.csv', index=False)
print("Merged results saved to timing_comparison.csv")

# Load the merged data
df = pd.read_csv("timing_comparison.csv")

# Plot settings
bar_width = 0.35
index = range(len(df))

plt.figure(figsize=(10, 6))

# Plot Classic bars
bars_classic = plt.bar(index, df['Time_Classic'], bar_width, label='Classic', color='steelblue')

# Plot Unified bars
bars_unified = plt.bar([i + bar_width for i in index], df['Time_Unified'], bar_width, label='Unified Memory', color='darkorange')

# Add value labels on top of each bar
for bar in bars_classic:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, label, ha='center', va='bottom', fontsize=8)

for bar in bars_unified:
    yval = bar.get_height()
    label = f'{yval:.3f}'.rstrip('0').rstrip('.')
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, label, ha='center', va='bottom', fontsize=8)

# Labels and ticks
plt.xlabel('Operation')
plt.ylabel('Time (ms)')
plt.title('Performance Comparison: Classic vs Unified Memory')
plt.xticks([i + bar_width / 2 for i in index], df['Operation'], rotation=15)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save or show
plt.savefig("comparison_plot.png")
plt.show()
