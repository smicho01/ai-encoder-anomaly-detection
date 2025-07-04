import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Generate some dummy normal data
X_train = np.random.normal(0, 1, (1000, 10))  # 1000 samples, 10 features

# Build autoencoder
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(6, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train only on normal data
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True)

# Test on some new data (with injected anomalies)
X_test = np.random.normal(0, 1, (200, 10))
X_test[0] += 10  # Inject anomaly into first row

# Predict & compute reconstruction error
X_pred = autoencoder.predict(X_test)
errors = np.mean((X_pred - X_test) ** 2, axis=1)

# Thresholding
threshold = np.percentile(errors, 95)  # top 5% considered anomalies
anomalies = errors > threshold
print("Anomalies Detected:", np.where(anomalies)[0])


for i, err in enumerate(errors):
    print(f"Sample {i}: Error = {err:.4f} {' <-- Anomaly' if anomalies[i] else ''}")


import matplotlib.pyplot as plt

plt.hist(errors, bins=50)
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()

