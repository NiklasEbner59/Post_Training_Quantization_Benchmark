import os
import pandas as pd
import matplotlib.pyplot as plt

# Consistent styling for all plots
plt.style.use('seaborn-v0_8-whitegrid')

# Base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load benchmark data for both models
file_i8 = os.path.join(BASE_DIR, 'benchmark_int8.csv')
file_f32 = os.path.join(BASE_DIR, 'benchmark_float32.csv')

df_i8 = pd.read_csv(file_i8)
df_f32 = pd.read_csv(file_f32)

# Convert latency from nanoseconds to milliseconds for better readability
df_i8['Latency_ms'] = df_i8['Latency_ns'] / 1_000_000.0
df_f32['Latency_ms'] = df_f32['Latency_ns'] / 1_000_000.0

# Calculate elapsed time in seconds from the initial timestamp for both datasets
df_i8['Time_s'] = (df_i8['Timestamp_ms'] - df_i8['Timestamp_ms'].iloc[0]) / 1000.0
df_f32['Time_s'] = (df_f32['Timestamp_ms'] - df_f32['Timestamp_ms'].iloc[0]) / 1000.0

# Round temperature values to 1 decimal place for cleaner visualization
df_i8['Time_s_int'] = df_i8['Time_s'].round().astype(int)
df_f32['Time_s_int'] = df_f32['Time_s'].round().astype(int)

temp_i8_mean = df_i8.groupby('Time_s_int')['Temp_C'].mean()
temp_f32_mean = df_f32.groupby('Time_s_int')['Temp_C'].mean()

# Global styling parameters for all plots
colors = ['#e74c3c', '#2ecc71']
labels = ['Float32 (Original)', 'Int8 (Quantized)']
bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.5, alpha=0.9)


# ==============================================================================
# DIAGRAM 1: TEMPERATURE TREND
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(temp_f32_mean.index / 60.0, temp_f32_mean.values,
        label='Float32: Baseline Thermal Footprint', color='#e74c3c', linewidth=2.5)
ax.plot(temp_i8_mean.index / 60.0, temp_i8_mean.values,
        label='Int8: Reduced Thermal Footprint', color='#2ecc71', linewidth=2.5)

ax.set_xlabel('Elapsed Time (Minutes)', fontsize=12, fontweight='bold')
ax.set_ylabel('System Temperature (°C)', fontsize=12, fontweight='bold')
ax.set_title('Thermal Behavior During 10-Minute Stress Test', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

# Scaling the axes to focus on the relevant temperature range and time window
ax.set_xlim(0, 10.1)
ax.set_ylim(26.5, 32)

# Calculating key temperature metrics for the annotation box
start_temp = temp_f32_mean.iloc[0]
max_f32 = temp_f32_mean.max()
max_i8 = temp_i8_mean.max()
temp_delta = max_f32 - max_i8

text_note = (f"Baseline Start Temp: {start_temp:.1f}°C\n"
             f"ΔT Float32 (Max): +{max_f32 - start_temp:.1f}°C\n"
             f"ΔT Int8 (Max): +{max_i8 - start_temp:.1f}°C\n"
             f"Thermal Saving: -{temp_delta:.1f}°C")

ax.text(9.9, 26.8, text_note, ha="right", va="bottom", size=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'final_temperature_trend.png'), dpi=300)
plt.close()


# ==============================================================================
# DIAGRAM 2: LATENCY COMPARISON
# ==============================================================================
mean_f32 = df_f32['Latency_ms'].mean()
mean_i8 = df_i8['Latency_ms'].mean()
stds = [df_f32['Latency_ms'].std(), df_i8['Latency_ms'].std()]
means = [mean_f32, mean_i8]
speedup = mean_f32 / mean_i8

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, means, color=colors, edgecolor='black', alpha=0.9)

ax.set_ylabel('Inference Latency (ms)', fontsize=12, fontweight='bold')
ax.set_title('MobileNetV2: Inference Latency Performance Gain', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 25)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Place exact latency values and standard deviation on top of each bar
for bar, std in zip(bars, stds):
    height = bar.get_height()
    ax.annotate(f'{height:.2f} ms\n(σ: {std:.2f} ms)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Adding the speedup box annotation
ax.text(0.5, 18, f"Speedup:\nx{speedup:.1f}", ha="center", va="center", size=13,
        fontweight='bold', bbox=bbox_props)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'final_latency_comparison.png'), dpi=300)
plt.close()


# ==============================================================================
# DIAGRAM 3: THROUGHPUT COMPARISON
# ==============================================================================
count_f32 = len(df_f32)
count_i8 = len(df_i8)
counts = [count_f32, count_i8]
pct_increase = ((count_i8 - count_f32) / count_f32) * 100

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.9)

ax.set_ylabel('Total Inferences (Count)', fontsize=12, fontweight='bold')
ax.set_title('MobileNetV2: Total Executions within 10-Minute Window', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 105000)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Format y-axis with commas for thousands
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Write the exact count values on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Adding the percentage increase box annotation
ax.text(0.5, 75000, f"Throughput Increase:\n+{pct_increase:.1f}%", ha="center", va="center", size=13,
        fontweight='bold', bbox=bbox_props)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'final_throughput_comparison.png'), dpi=300)
plt.close()

print("All 3 publication-quality plots successfully created in the script directory!")