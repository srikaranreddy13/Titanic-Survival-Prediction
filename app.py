import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Load model weights and scaler (no TensorFlow needed)
# ---------------------------------------------------
model_data = joblib.load('titanic_weights.pkl')
scaler = joblib.load('scaler.pkl')

def predict(X):
    """Pure numpy forward pass: ReLU -> ReLU -> Sigmoid"""
    h1 = np.maximum(0, X @ model_data['W1'] + model_data['b1'])
    h2 = np.maximum(0, h1 @ model_data['W2'] + model_data['b2'])
    out = 1 / (1 + np.exp(-(h2 @ model_data['W3'] + model_data['b3'])))
    return out

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; }
    .title { font-size:40px; color:white; font-weight:bold; }
    .subtitle { font-size:20px; color:lightgray; }
    .card { background-color:#1E1E1E; padding:20px; border-radius:15px; color:white; }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🚢")
with col2:
    st.markdown('<div class="title">Titanic Survival Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Deep Learning Based Passenger Survival Prediction</div>', unsafe_allow_html=True)

st.write("---")
st.markdown("## Project Description")
st.markdown(
    """
This application predicts whether a passenger would survive during the Titanic disaster using an Artificial Neural Network (ANN).

The system is developed using:

- TensorFlow / Keras (trained)
- Deep Learning
- Streamlit Deployment
- Min-Max Normalization

The trained ANN model performs real-time inference based on passenger details.
"""
)

st.write("---")
st.markdown("## Passenger Input Form")

col1, col2, col3 = st.columns(3)
with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
with col2:
    age = st.slider("Age", 1, 80, 25)
with col3:
    fare = st.number_input("Fare", min_value=0.0, value=50.0)

st.write("---")

predict_btn = st.button("Predict Survival")

if predict_btn:
    input_df = pd.DataFrame({'Pclass': [pclass], 'Age': [age], 'Fare': [fare]})
    input_scaled = scaler.transform(input_df)
    probability = float(predict(input_scaled)[0][0])

    if probability > 0.5:
        result = "Survived"
        confidence = probability * 100
    else:
        result = "Not Survived"
        confidence = (1 - probability) * 100

    st.write("---")
    st.markdown("## Prediction Output")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", result)
    with col2:
        st.metric("Survival Probability", f"{probability:.2f}")
    with col3:
        st.metric("Confidence Score", f"{confidence:.2f}%")

    st.write("---")
    st.markdown("## Survival Probability Visualization")

    chart_data = pd.DataFrame({
        'Category': ['Survived', 'Not Survived'],
        'Probability': [probability, 1 - probability]
    })
    st.bar_chart(chart_data.set_index('Category'))

    fig, ax = plt.subplots()
    ax.pie(
        [probability, 1 - probability],
        labels=['Survived', 'Not Survived'],
        autopct='%1.1f%%'
    )
    st.pyplot(fig)