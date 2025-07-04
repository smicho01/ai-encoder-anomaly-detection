import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load dataset
df = pd.read_csv("mock_transactions.csv")

# Separate labels
labels = df["Label"]
df = df.drop("Label", axis=1)

# Identify numerical and categorical features
num_features = ["Amount", "Hour"]
cat_features = ["Country", "Merchant_Type"]

# Preprocess: scale numeric, one-hot encode categorical
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(sparse_output=False), cat_features)
])

X_processed = preprocessor.fit_transform(df)

# Use only normal data (label == 0) for training
X_train = X_processed[labels == 0]

# Build autoencoder
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)
decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, shuffle=True, verbose=1)

# Predict on full dataset
X_pred = autoencoder.predict(X_processed)
errors = np.mean((X_pred - X_processed) ** 2, axis=1)

# Use 95th percentile of training errors as threshold
threshold = np.percentile(errors[labels == 0], 95)
predicted_anomalies = errors > threshold

# Compute evaluation metrics
true_positives = np.sum((predicted_anomalies == 1) & (labels == 1))
false_positives = np.sum((predicted_anomalies == 1) & (labels == 0))
false_negatives = np.sum((predicted_anomalies == 0) & (labels == 1))
true_negatives = np.sum((predicted_anomalies == 0) & (labels == 0))

metrics = {
    "Threshold": threshold,
    "True Positives": int(true_positives),
    "False Positives": int(false_positives),
    "False Negatives": int(false_negatives),
    "True Negatives": int(true_negatives),
    "Accuracy": float((true_positives + true_negatives) / len(labels))
}

# Print metrics
print("\n--- Detection Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Show predicted anomalies
print("\n--- Detected Anomalies ---")
anomaly_indices = np.where(predicted_anomalies)[0]
print(df.iloc[anomaly_indices])

# Show all actual anomalies
print("\n--- All Actual Anomalies (Label = 1) ---")
actual_anomalies_df = df[labels == 1]
print(actual_anomalies_df)

# Show false positives (model thought it was anomaly, but it's normal)
print("\n--- False Positives (Predicted=1, Label=0) ---")
fp_df = df[(predicted_anomalies == 1) & (labels == 0)]
print(fp_df)

# Show false negatives (model missed the anomaly)
print("\n--- False Negatives (Predicted=0, Label=1) ---")
fn_df = df[(predicted_anomalies == 0) & (labels == 1)]
print(fn_df)

# Plot reconstruction error distribution
plt.figure(figsize=(10, 5))
plt.hist(errors[labels == 0], bins=50, alpha=0.6, label="Normal")
plt.hist(errors[labels == 1], bins=50, alpha=0.6, label="Anomaly")
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("reconstruction_error.png")
print("Plot saved to reconstruction_error.png")