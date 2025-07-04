#  To run app use command: streamlit run 04_b_anomaly_Streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from tensorflow.keras.models import load_model

import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError


def train_autoencoder(data_path, epochs, batch_size, threshold_percentile):
    # Load dataset
    df = pd.read_csv(data_path)
    labels = df["Label"]
    df = df.drop("Label", axis=1)

    # Identify features
    num_features = ["Amount", "Hour"]
    cat_features = ["Country", "Merchant_Type"]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(sparse_output=False), cat_features)
    ])
    X_processed = preprocessor.fit_transform(df)
    X_train = X_processed[labels == 0]  # only normal

    # Autoencoder
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded)
    decoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

    # Compute threshold
    X_pred = autoencoder.predict(X_processed)
    errors = np.mean((X_pred - X_processed) ** 2, axis=1)
    threshold = float(np.percentile(errors[labels == 0], threshold_percentile))

    # Save artifacts
    autoencoder.save("model.keras")
    joblib.dump(preprocessor, "preprocessor.pkl")
    with open("threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    return threshold, preprocessor, df

st.title("🧠 Anomaly Detection on Transactions")

# --- Part 1: Training Section ---
st.header("1️⃣ Train Autoencoder")

with st.form("train_form"):
    data_path = st.text_input("CSV File Path", value="mock_transactions.csv")
    epochs = st.slider("Epochs", 10, 200, 50)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)
    threshold_percentile = st.slider("Threshold Percentile", 90, 99, 95)
    submitted = st.form_submit_button("🚀 Train Model")

if submitted:
    with st.spinner("Training in progress..."):
        threshold, preprocessor, df = train_autoencoder(data_path, epochs, batch_size, threshold_percentile)
        st.success(f"Model trained and saved. Threshold set to: {threshold:.5f}")
        st.session_state.trained = True
        st.session_state.categories = {
            "Country": sorted(df["Country"].unique()),
            "Merchant_Type": sorted(df["Merchant_Type"].unique())
        }

# --- Part 2: Detection Section ---
if "trained" in st.session_state and st.session_state.trained:
    st.header("2️⃣ Check a Transaction")

    with st.form("anomaly_form"):
        amount = st.number_input("Amount", value=50.0, min_value=0.0)
        hour = st.slider("Hour", 0, 23, 12)
        country = st.selectbox("Country", st.session_state.categories["Country"])
        merchant_type = st.selectbox("Merchant Type", st.session_state.categories["Merchant_Type"])
        detect = st.form_submit_button("🔍 Detect Anomaly")

    if detect:
        # Load model and preprocessor
        model = load_model("model.keras")
        preprocessor = joblib.load("preprocessor.pkl")
        with open("threshold.json", "r") as f:
            threshold = json.load(f)["threshold"]

        # Build input
        tx = {
            "Amount": amount,
            "Country": country,
            "Merchant_Type": merchant_type,
            "Hour": hour
        }
        df_input = pd.DataFrame([tx])
        X_input = preprocessor.transform(df_input)
        X_pred = model.predict(X_input)
        error = float(np.mean((X_pred - X_input) ** 2))
        is_anomaly = error > threshold

        # Show result
        st.markdown("### 🧾 Transaction Summary")
        st.json(tx)

        st.markdown("### 🔎 Detection Result")
        st.write(f"Reconstruction Error: `{error:.5f}`")
        st.write(f"Threshold: `{threshold:.5f}`")
        if is_anomaly:
            st.error("⚠️ This transaction is an **ANOMALY**.")
        else:
            st.success("✅ This transaction is **normal**.")
