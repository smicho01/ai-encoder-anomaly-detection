# 📊 Bank Transaction Anomaly Detection (Autoencoder Neural Network)

This project uses a **Neural Network (Autoencoder)** to detect **anomalies (e.g. fraud)** in bank transaction data.

It simulates how a bank might monitor user behavior to detect unusual activity such as:

- Unusually large purchases
- Transactions from foreign countries
- Spending at odd hours

---

## 🧠 How It Works

This script uses an **autoencoder**, which is a special type of neural network designed for **unsupervised learning**. It works in two phases:

### 1. Training Phase

- Trained **only on normal (non-anomalous)** transactions
- Learns to **reconstruct normal behavior** by identifying patterns in typical transaction amounts, countries, merchant types, and times

### 2. Anomaly Detection Phase

- When fed new data, it attempts to reconstruct each transaction
- If the reconstruction error is **high**, it suggests the transaction **doesn’t fit the learned pattern** — and is likely an anomaly

---

## 💡 Why Use a Neural Network?

Unlike manual rule-based systems (e.g., “flag if amount > $1000”), a neural network:

- Understands **complex patterns** in multidimensional data
- Adapts to the **specific behavior** of a user
- Can generalize to new, unseen patterns of normal behavior

This is especially useful when anomalies are rare or hard to define explicitly.

---

## 🗂️ What the Script Does

1. Loads mock transaction data (`mock_transactions.csv`)
2. Preprocesses it (scales numeric features, encodes categorical ones)
3. Builds and trains an autoencoder neural network using **Keras/TensorFlow**
4. Calculates reconstruction error for all transactions
5. Flags those with high error as anomalies
6. Prints:
   - Detected anomalies
   - Evaluation metrics (true/false positives/negatives)
7. Plots the distribution of reconstruction errors

---

## 🧪 Sample Features in the Dataset

| Feature         | Description                              |
| --------------- | ---------------------------------------- |
| `Amount`        | Transaction amount                       |
| `Country`       | Country where the transaction occurred   |
| `Merchant_Type` | Type of merchant (e.g. Grocery, Crypto)  |
| `Hour`          | Hour of the day the transaction occurred |

---

## What is Reconstruction Error?

Reconstruction error is a measure of how well the autoencoder neural network can recreate (or "reconstruct") the input data.
When the model is

- trained only on normal (non-anomalous) data, it learns patterns of what "normal" looks like.
- Later, when it tries to reconstruct new data, it performs well on similar (normal) data and poorly on different (anomalous) data.
- The difference between the original and reconstructed data is called the reconstruction error (often calculated using Mean Squared Error, or MSE).

In short:

- Low error = looks normal
- High error = likely anomaly

## 🧰 Setup Instructions

### 1. Create a Python virtual environment

```bash
python -m venv env
```

### 2. Activate the environment

- On macOS/Linux:
  ```bash
  source env/bin/activate
  ```
- On Windows:
  ```bash
  env\Scripts\activate.bat
  ```

### 3. Install required dependencies

Create a `requirements.txt` file with:

```txt
tensorflow
pandas
numpy
matplotlib
scikit-learn
```

Then install them:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Script

Make sure `mock_transactions.csv` is in the same directory, then run:

```bash
python anomaly_bank_transaction.py
```

You'll see printed outputs and a plot showing how anomalies differ from normal transactions.

---

## 📝 Output Example

```
--- Detection Metrics ---
Threshold: 0.1141
True Positives: 50
False Positives: 48
False Negatives: 0
Accuracy: 95.2%

--- Detected Anomalies ---
[...list of suspicious transactions...]
```

---

Let me know if you'd like a version for real-time monitoring, a Streamlit dashboard, or CSV output!
