import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate normal transactions
num_normal = 950
normal_data = {
    "Amount": np.random.normal(50, 20, num_normal).clip(5, 200),  # mostly small purchases
    "Country": ["UK"] * num_normal,
    "Merchant_Type": np.random.choice(["Grocery", "ATM", "Utilities", "Restaurant", "Online"], num_normal),
    "Hour": np.random.randint(6, 22, num_normal)  # normal daytime hours
}

normal_df = pd.DataFrame(normal_data)

# Generate anomalous transactions
num_anomalies = 50
anomalous_data = {
    "Amount": np.random.uniform(500, 5000, num_anomalies),  # unusually high
    "Country": np.random.choice(["UAE", "Mexico", "USA", "Thailand", "Unknown"], num_anomalies),
    "Merchant_Type": np.random.choice(["Jewelry", "Crypto", "Gambling", "Luxury"], num_anomalies),
    "Hour": np.random.choice([1, 2, 3, 4], num_anomalies)  # late-night activity
}

anomalous_df = pd.DataFrame(anomalous_data)

# Combine into a single dataset
df = pd.concat([normal_df, anomalous_df], ignore_index=True)
df["Label"] = [0] * num_normal + [1] * num_anomalies  # 0 = normal, 1 = anomaly

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
file_path = "/mnt/data/mock_transactions.csv"
df.to_csv(file_path, index=False)

file_path
