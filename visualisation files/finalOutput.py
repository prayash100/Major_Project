import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("GA_output.csv")

# Plot 1: Total Cost as a Point Plot
plt.figure(figsize=(12, 6))
plt.plot(df["Index"], df["total_cost"], 'o', markersize=3, color='darkorange')
plt.title("Total Cost per Index (Point Plot)")
plt.xlabel("Index")
plt.ylabel("Total Cost")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot both columns
plt.figure(figsize=(10, 6))
plt.plot(df['gas_needed'], 'o', markersize=3, color='red')
plt.plot(df['gas_availability'], label='Wind Available', color='blue')
plt.xlabel('Time')
plt.ylabel('Wind Power (MW)')  # Change unit if needed
plt.title('Wind Available vs Wind Needed')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Frequency Mean with Tight Y-Scale
plt.figure(figsize=(12, 6))
#plt.plot(df["Index"], df["frequency_mean"], label="Mean", color='blue')
plt.plot(df["Index"], df["frequency_std_dev"], label="Std Dev", color='green')
plt.plot(df["Index"], df["frequency_max_deviation"], label="Max Dev", color='red')
plt.plot(df["Index"], df["frequency_min_deviation"], label="Min Dev", color='purple')
plt.title("Frequency Metrics Over Time (Zoomed)")
plt.xlabel("Index")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Load the full dataset
data = np.load("GA_frequency_data.npy")  # shape: (864, 100)

# Flatten to 1D array of 86400 values
all_freqs = data.flatten()  # shape: (86400,)

# Plot all values
plt.figure(figsize=(16, 5))  # wider figure for readability
plt.plot(all_freqs, linewidth=0.5)
plt.xlabel("Time Step")
plt.ylabel("Frequency (Hz)")
plt.title("Frequency Variation Over Time (86400 Time Steps)")
plt.grid(True)
plt.tight_layout()
plt.show()
