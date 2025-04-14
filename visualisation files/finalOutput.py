import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Load your CSV file
df = pd.read_csv("BSA_output.csv")  # Update with the correct path if needed

# Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df["Index"], y=df["total_cost"], color="orange", label="Total Cost by BSA", alpha=0.5)
sns.lineplot(x=df["Index"], y=df["total_cost"].rolling(window=3, center=True).mean(), color="black", linewidth=2, label="Guided Trend (5 minute)")

plt.title("Total Cost Over Time with BSA")
plt.xlabel("Index (Time blocks 100s)")
plt.ylabel("Total Cost( *1e6 INR)")
plt.legend(title="BSA Action / Trend")
plt.grid(True)
plt.tight_layout()
plt.show()

# Load the .npy frequency file
freq = np.load("BSA_frequency_data.npy")  # Change to your file path if needed

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

plt.title("Frequency Over Time by BSA")
plt.xlabel("Time Steps (second)")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

