from tensorflow.keras.models import load_model
import joblib
import numpy as np
import json

# Load threshold from JSON file
with open("threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# Load model and preprocessor
autoencoder = load_model("anomaly_autoencoder.keras")
preprocessor = joblib.load("preprocessor.pkl")


def is_anomaly(transaction: dict, model, preprocessor, threshold: float) -> bool:
    import pandas as pd
    df_input = pd.DataFrame([transaction])
    X_input = preprocessor.transform(df_input)
    X_pred = model.predict(X_input)
    error = np.mean((X_pred - X_input) ** 2)
    return error > threshold


txn1 = {
    "Amount": 20.0,
    "Country": "UK",
    "Merchant_Type": "Grocery",
    "Hour": 15
}

txn2 = {
    "Amount": 2000.0,
    "Country": "UK",
    "Merchant_Type": "Grocery",
    "Hour": 15
}

result1 = is_anomaly(txn1, autoencoder, preprocessor, threshold)
result2 = is_anomaly(txn2, autoencoder, preprocessor, threshold)

print("Txn 1 - Anomaly?" , result1)
print("Txn 2 - Anomaly?" , result2)