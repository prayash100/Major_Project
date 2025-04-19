import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the CSV file
df = pd.read_csv("new/sa/sa_new.csv")

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
plt.plot(df['gas_availability'], label='Gas Available', color='blue')
plt.xlabel('Time')
plt.ylabel('Gas Power (MW)')  # Change unit if needed
plt.title('Gas Available vs Gas Needed')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['coal_needed'], 'o', markersize=3, color='red')
plt.plot(df['coal_availability'], label='Coal Available', color='blue')
plt.xlabel('Time')
plt.ylabel('coal Power (MW)')  # Change unit if needed
plt.title('Coal Available vs Coal Needed')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['wind_needed'], 'o', markersize=3, color='red')
plt.plot(df['wind_availability'], label='Wind Available', color='blue')
plt.xlabel('Time')
plt.ylabel('Wind Power (MW)')  # Change unit if needed
plt.title('Wind Available vs Wind Needed')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['solar_needed'], 'o', markersize=3, color='red')
plt.plot(df['solar_availability'], label='Solar Available', color='blue')
plt.xlabel('Time')
plt.ylabel('Solar Power (MW)')  # Change unit if needed
plt.title('Solar Available vs Solar Needed')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Load the full dataset
freq = np.load("new/sa/freq/sa_frequency.npy")  # Change to your file path if needed

# Flatten if it's not already 1D
if freq.ndim > 1:
    freq = freq.flatten()

# Create a DataFrame
df_freq = pd.DataFrame({
    "Index": np.arange(len(freq)),
    "frequency": freq
})

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_freq, x="Index", y="frequency", color="steelblue", label="Frequency", linewidth=1)
sns.lineplot(data=df_freq, x="Index", y=df_freq["frequency"].rolling(window=300, center=True).mean(),
             color="black", linewidth=2, label="Guided Trend (5 minute)")

plt.title("Frequency Over Time by sa")
plt.xlabel("Time Steps (second)")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()