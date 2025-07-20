# Interactive Autoencoder Demo for AI Presentation
# Run with: streamlit run interactive_autoencoder_demo.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import Callback
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import io


# Custom callback to capture training progress
class TrainingProgressCallback(Callback):
    def __init__(self):
        self.losses = []
        self.epochs_completed = 0

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss', 0))
        self.epochs_completed = epoch + 1


# Set page config
st.set_page_config(
    page_title="🧠 AI Autoencoder Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Interactive Autoencoder Demo")
st.markdown("### Understanding Neural Networks and Anomaly Detection")

# Sidebar for navigation
st.sidebar.title("📚 Demo Sections")
demo_section = st.sidebar.radio(
    "Choose a section:",
    ["🎯 What is an Autoencoder?",
     "🏗️ Architecture Visualization",
     "📊 Generate Training Data",
     "🚀 Train the Model (Live)",
     "🔍 Anomaly Detection",
     "📈 Model Analysis"]
)

# Section 1: What is an Autoencoder?
if demo_section == "🎯 What is an Autoencoder?":
    st.header("🎯 What is an Autoencoder?")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        An **Autoencoder** is a type of neural network that learns to compress and reconstruct data.

        ### Key Concepts:
        - **Encoder**: Compresses input data into a smaller representation
        - **Decoder**: Reconstructs the original data from the compressed version
        - **Bottleneck**: The smallest layer that forces the network to learn important features
        - **Reconstruction Error**: How different the output is from the input

        ### For Anomaly Detection:
        - Train only on "normal" data
        - Normal data reconstructs well (low error)
        - Anomalous data reconstructs poorly (high error)
        - Set a threshold to classify anomalies
        """)

    with col2:
        # Simple autoencoder diagram
        fig = go.Figure()

        # Input layer
        fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=4,
                      fillcolor="lightblue", line=dict(color="blue"))
        fig.add_annotation(x=0.5, y=2, text="Input<br>Layer<br>(4 features)", showarrow=False)

        # Hidden layer 1
        fig.add_shape(type="rect", x0=2, y0=0.5, x1=3, y1=3.5,
                      fillcolor="lightgreen", line=dict(color="green"))
        fig.add_annotation(x=2.5, y=2, text="Hidden<br>Layer<br>(3 neurons)", showarrow=False)

        # Bottleneck
        fig.add_shape(type="rect", x0=4, y0=1, x1=5, y1=3,
                      fillcolor="orange", line=dict(color="red"))
        fig.add_annotation(x=4.5, y=2, text="Bottleneck<br>(2 neurons)", showarrow=False)

        # Hidden layer 2
        fig.add_shape(type="rect", x0=6, y0=0.5, x1=7, y1=3.5,
                      fillcolor="lightgreen", line=dict(color="green"))
        fig.add_annotation(x=6.5, y=2, text="Hidden<br>Layer<br>(3 neurons)", showarrow=False)

        # Output layer
        fig.add_shape(type="rect", x0=8, y0=0, x1=9, y1=4,
                      fillcolor="lightcoral", line=dict(color="red"))
        fig.add_annotation(x=8.5, y=2, text="Output<br>Layer<br>(4 features)", showarrow=False)

        # Arrows
        for start, end in [(1, 2), (3, 4), (5, 6), (7, 8)]:
            fig.add_annotation(x=start + 0.5, y=2, ax=start, ay=2,
                               arrowhead=2, arrowsize=1, arrowwidth=2)

        fig.update_layout(
            title="Autoencoder Architecture",
            xaxis=dict(range=[-0.5, 9.5], showticklabels=False),
            yaxis=dict(range=[-0.5, 4.5], showticklabels=False),
            height=300,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

# Section 2: Architecture Visualization
elif demo_section == "🏗️ Architecture Visualization":
    st.header("🏗️ Autoencoder Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎛️ Design Your Architecture")
        input_dim = st.slider("Input Dimension", 2, 20, 8)
        hidden_dim1 = st.slider("First Hidden Layer", 2, input_dim - 1, 6)
        bottleneck_dim = st.slider("Bottleneck Dimension", 1, hidden_dim1 - 1, 3)
        activation = st.selectbox("Activation Function", ["relu", "tanh", "sigmoid"])

        st.markdown("### 📋 Architecture Summary:")
        st.code(f"""
Input Layer:     {input_dim} neurons
Hidden Layer 1:  {hidden_dim1} neurons ({activation})
Bottleneck:      {bottleneck_dim} neurons ({activation})
Hidden Layer 2:  {hidden_dim1} neurons ({activation})
Output Layer:    {input_dim} neurons (sigmoid)
        """)

    with col2:
        st.subheader("🏗️ Network Visualization")

        # Create network visualization
        layers = [input_dim, hidden_dim1, bottleneck_dim, hidden_dim1, input_dim]
        layer_names = ["Input", "Encoder 1", "Bottleneck", "Decoder 1", "Output"]
        colors = ["lightblue", "lightgreen", "orange", "lightgreen", "lightcoral"]

        fig = go.Figure()

        for i, (size, name, color) in enumerate(zip(layers, layer_names, colors)):
            x_pos = i * 2
            y_positions = np.linspace(-size / 2, size / 2, size)

            for j, y_pos in enumerate(y_positions):
                fig.add_trace(go.Scatter(
                    x=[x_pos], y=[y_pos],
                    mode='markers',
                    marker=dict(size=20, color=color, line=dict(width=2, color='black')),
                    showlegend=False,
                    hovertemplate=f"{name}<br>Neuron {j + 1}<extra></extra>"
                ))

            # Add layer labels
            fig.add_annotation(
                x=x_pos, y=max(y_positions) + 1,
                text=f"{name}<br>({size})",
                showarrow=False,
                font=dict(size=10)
            )

            # Add connections to next layer
            if i < len(layers) - 1:
                next_y_positions = np.linspace(-layers[i + 1] / 2, layers[i + 1] / 2, layers[i + 1])
                for y1 in y_positions:
                    for y2 in next_y_positions:
                        fig.add_shape(
                            type="line",
                            x0=x_pos + 0.1, y0=y1,
                            x1=x_pos + 1.9, y1=y2,
                            line=dict(color="gray", width=0.5, dash="dot")
                        )

        fig.update_layout(
            title="Neural Network Architecture",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            height=400,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

# Section 3: Generate Training Data
elif demo_section == "📊 Generate Training Data":
    st.header("📊 Generate Training Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎛️ Data Parameters")
        n_normal = st.slider("Number of Normal Samples", 100, 1000, 500)
        n_anomaly = st.slider("Number of Anomaly Samples", 10, 100, 50)
        noise_level = st.slider("Noise Level", 0.1, 2.0, 0.5)
        random_seed = st.slider("Random Seed", 1, 100, 42)

        if st.button("🎲 Generate New Dataset"):
            # Generate normal data (circular pattern)
            np.random.seed(random_seed)
            angle = np.random.uniform(0, 2 * np.pi, n_normal)
            radius = np.random.normal(5, 1, n_normal)
            normal_x = radius * np.cos(angle) + np.random.normal(0, noise_level, n_normal)
            normal_y = radius * np.sin(angle) + np.random.normal(0, noise_level, n_normal)

            # Generate anomaly data (scattered points)
            anomaly_x = np.random.uniform(-15, 15, n_anomaly)
            anomaly_y = np.random.uniform(-15, 15, n_anomaly)

            # Create DataFrame
            normal_df = pd.DataFrame({
                'x': normal_x, 'y': normal_y,
                'amount': np.random.exponential(50, n_normal),
                'hour': np.random.randint(0, 24, n_normal),
                'label': 0
            })

            anomaly_df = pd.DataFrame({
                'x': anomaly_x, 'y': anomaly_y,
                'amount': np.random.exponential(200, n_anomaly),
                'hour': np.random.randint(0, 24, n_anomaly),
                'label': 1
            })

            df = pd.concat([normal_df, anomaly_df], ignore_index=True)
            st.session_state.training_data = df

    with col2:
        if 'training_data' in st.session_state:
            st.subheader("📈 Data Visualization")
            df = st.session_state.training_data

            fig = px.scatter(
                df, x='x', y='y', color='label',
                color_discrete_map={0: 'blue', 1: 'red'},
                labels={'label': 'Type'},
                title="Generated Dataset"
            )
            fig.update_traces(
                marker=dict(size=8),
                selector=dict(name="0")
            )
            fig.update_traces(
                marker=dict(size=12, symbol='x'),
                selector=dict(name="1")
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Data Statistics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Normal Samples", len(df[df.label == 0]))
                st.metric("Mean Amount (Normal)", f"${df[df.label == 0]['amount'].mean():.2f}")
            with col_b:
                st.metric("Anomaly Samples", len(df[df.label == 1]))
                st.metric("Mean Amount (Anomaly)", f"${df[df.label == 1]['amount'].mean():.2f}")

# Section 4: Train the Model (Live)
elif demo_section == "🚀 Train the Model (Live)":
    st.header("🚀 Live Model Training")

    if 'training_data' not in st.session_state:
        st.warning("⚠️ Please generate training data first in the previous section!")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎛️ Training Parameters")
        epochs = st.slider("Number of Epochs", 10, 100, 50)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)

        if st.button("🚀 Start Training"):
            df = st.session_state.training_data

            # Prepare data (only normal samples for training)
            normal_data = df[df.label == 0][['x', 'y', 'amount', 'hour']].values
            scaler = StandardScaler()
            X_train = scaler.fit_transform(normal_data)

            # Build model
            input_dim = X_train.shape[1]
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(3, activation='relu')(input_layer)
            encoded = Dense(2, activation='relu')(encoded)
            decoded = Dense(3, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='sigmoid')(decoded)

            autoencoder = Model(inputs=input_layer, outputs=decoded)
            autoencoder.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse'
            )

            # Create progress tracking
            progress_callback = TrainingProgressCallback()

            # Training progress containers
            progress_bar = st.progress(0)
            loss_placeholder = st.empty()
            epoch_placeholder = st.empty()

            # Train with live updates
            history = autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[progress_callback],
                verbose=0
            )

            # Update progress
            for epoch in range(epochs):
                progress_bar.progress((epoch + 1) / epochs)
                epoch_placeholder.text(f"Epoch: {epoch + 1}/{epochs}")
                if epoch < len(history.history['loss']):
                    loss_placeholder.text(f"Loss: {history.history['loss'][epoch]:.6f}")
                time.sleep(0.1)  # Simulate real-time training

            # Store results
            st.session_state.trained_model = autoencoder
            st.session_state.scaler = scaler
            st.session_state.training_history = history.history

            st.success("✅ Training completed!")

    with col2:
        if 'training_history' in st.session_state:
            st.subheader("📈 Training Progress")

            history = st.session_state.training_history

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=3)
            ))
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red', width=3)
                ))

            fig.update_layout(
                title="Training Loss Over Time",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

            # Model summary
            st.subheader("🏗️ Model Architecture")
            if 'trained_model' in st.session_state:
                model = st.session_state.trained_model

                # Capture model summary
                stringlist = []
                model.summary(print_fn=lambda x: stringlist.append(x))
                summary_string = "\n".join(stringlist)
                st.code(summary_string)

            st.markdown("(None, 4) - What is 'None' ? - batch dimension.  it means 'any number of samples' or 'flexible batch size'")
            st.markdown("Input shape: (100, 4)    # 100 transactions, each with 4 features")

# Section 5: Anomaly Detection
elif demo_section == "🔍 Anomaly Detection":
    st.header("🔍 Real-time Anomaly Detection")

    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train the model first!")
        st.stop()

    model = st.session_state.trained_model
    scaler = st.session_state.scaler
    df = st.session_state.training_data

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎛️ Test New Data Point")

        test_x = st.slider("X coordinate", -15.0, 15.0, 0.0)
        test_y = st.slider("Y coordinate", -15.0, 15.0, 0.0)
        test_amount = st.slider("Amount ($)", 0.0, 500.0, 50.0)
        test_hour = st.slider("Hour", 0, 23, 12)

        threshold_percentile = st.slider("Anomaly Threshold (percentile)", 90, 99, 95)

        # Calculate threshold from training data
        normal_data = df[df.label == 0][['x', 'y', 'amount', 'hour']].values
        X_normal = scaler.transform(normal_data)
        normal_predictions = model.predict(X_normal, verbose=0)
        normal_errors = np.mean((normal_predictions - X_normal) ** 2, axis=1)
        threshold = np.percentile(normal_errors, threshold_percentile)

        # Test the new point
        test_point = np.array([[test_x, test_y, test_amount, test_hour]])
        test_scaled = scaler.transform(test_point)
        test_prediction = model.predict(test_scaled, verbose=0)
        test_error = np.mean((test_prediction - test_scaled) ** 2)

        is_anomaly = test_error > threshold

        st.subheader("🔎 Detection Result")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Reconstruction Error", f"{test_error:.6f}")
            st.metric("Threshold", f"{threshold:.6f}")
        with col_b:
            if is_anomaly:
                st.error("⚠️ ANOMALY DETECTED!")
                st.metric("Anomaly Score", f"{(test_error / threshold):.2f}x threshold")
            else:
                st.success("✅ Normal Transaction")
                st.metric("Anomaly Score", f"{(test_error / threshold):.2f}x threshold")

    with col2:
        st.subheader("📊 Visualization")

        # Create visualization showing the test point
        fig = px.scatter(
            df, x='x', y='y', color='label',
            color_discrete_map={0: 'lightblue', 1: 'lightcoral'},
            title="Data Distribution with Test Point"
        )

        # Add test point
        fig.add_trace(go.Scatter(
            x=[test_x], y=[test_y],
            mode='markers',
            marker=dict(
                size=20,
                color='red' if is_anomaly else 'green',
                symbol='star',
                line=dict(width=3, color='black')
            ),
            name='Test Point',
            showlegend=True
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Error distribution
        st.subheader("📈 Error Distribution")

        all_data = df[['x', 'y', 'amount', 'hour']].values
        X_all = scaler.transform(all_data)
        all_predictions = model.predict(X_all, verbose=0)
        all_errors = np.mean((all_predictions - X_all) ** 2, axis=1)

        fig2 = go.Figure()

        # Normal errors
        normal_mask = df.label == 0
        fig2.add_trace(go.Histogram(
            x=all_errors[normal_mask],
            name='Normal',
            opacity=0.7,
            nbinsx=30,
            marker_color='blue'
        ))

        # Anomaly errors
        anomaly_mask = df.label == 1
        fig2.add_trace(go.Histogram(
            x=all_errors[anomaly_mask],
            name='Anomaly',
            opacity=0.7,
            nbinsx=30,
            marker_color='red'
        ))

        # Add threshold line
        fig2.add_vline(x=threshold, line_dash="dash", line_color="green",
                       annotation_text="Threshold")

        # Add test point error
        fig2.add_vline(x=test_error, line_dash="dot", line_color="orange",
                       annotation_text="Test Point")

        fig2.update_layout(
            title="Reconstruction Error Distribution",
            xaxis_title="Reconstruction Error",
            yaxis_title="Frequency",
            barmode='overlay'
        )

        st.plotly_chart(fig2, use_container_width=True)

# Section 6: Model Analysis
elif demo_section == "📈 Model Analysis":
    st.header("📈 Model Analysis & Insights")

    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train the model first!")
        st.stop()

    model = st.session_state.trained_model
    scaler = st.session_state.scaler
    df = st.session_state.training_data

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Model Performance")

        # Calculate metrics
        all_data = df[['x', 'y', 'amount', 'hour']].values
        X_all = scaler.transform(all_data)
        all_predictions = model.predict(X_all, verbose=0)
        all_errors = np.mean((all_predictions - X_all) ** 2, axis=1)

        # Use 95th percentile as threshold
        threshold = np.percentile(all_errors[df.label == 0], 95)
        predicted_anomalies = all_errors > threshold
        true_anomalies = df.label == 1

        # Calculate confusion matrix
        tp = np.sum(predicted_anomalies & true_anomalies)
        fp = np.sum(predicted_anomalies & ~true_anomalies)
        tn = np.sum(~predicted_anomalies & ~true_anomalies)
        fn = np.sum(~predicted_anomalies & true_anomalies)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(df)

        # Display metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Precision", f"{precision:.3f}")
        with metric_col2:
            st.metric("Recall", f"{recall:.3f}")
            st.metric("F1-Score", f"{f1:.3f}")

        # Confusion Matrix
        st.subheader("🔄 Confusion Matrix")
        confusion_data = pd.DataFrame({
            'Predicted Normal': [tn, fn],
            'Predicted Anomaly': [fp, tp]
        }, index=['Actual Normal', 'Actual Anomaly'])

        st.dataframe(confusion_data, use_container_width=True)

    with col2:
        st.subheader("🧠 Feature Importance")

        # Analyze feature importance by perturbation
        feature_names = ['X', 'Y', 'Amount', 'Hour']
        importance_scores = []

        baseline_error = np.mean(all_errors[df.label == 0])

        for i in range(len(feature_names)):
            # Perturb feature
            X_perturbed = X_all.copy()
            X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])

            perturbed_predictions = model.predict(X_perturbed, verbose=0)
            perturbed_errors = np.mean((perturbed_predictions - X_perturbed) ** 2, axis=1)
            perturbed_error = np.mean(perturbed_errors[df.label == 0])

            importance = abs(perturbed_error - baseline_error)
            importance_scores.append(importance)

        # Plot feature importance
        fig = go.Figure(data=[
            go.Bar(x=feature_names, y=importance_scores,
                   marker_color=['blue', 'green', 'orange', 'red'])
        ])
        fig.update_layout(
            title="Feature Importance for Anomaly Detection",
            xaxis_title="Features",
            yaxis_title="Importance Score"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ROC Curve
        st.subheader("📊 ROC Curve")

        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(true_anomalies, all_errors)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))

        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=300
        )

        st.plotly_chart(fig_roc, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 🎓 Key Takeaways for Your Team:")
st.markdown("""
1. **Autoencoders learn patterns** in normal data without being explicitly told what's normal
2. **The bottleneck layer** forces the network to learn the most important features
3. **Reconstruction error** is the key metric - high error = potential anomaly
4. **Threshold tuning** is crucial for balancing false positives vs false negatives
5. **Feature engineering** can significantly impact model performance
""")
