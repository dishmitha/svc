import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="SVC Breast Cancer App",
    layout="wide"
)

st.title("Breast Cancer Classification using SVC")
st.write(
    "This app trains a Support Vector Classifier (SVC) model "
    "using the breast cancer dataset and predicts whether "
    "a tumor is Benign or Malignant."
)

# ---------------------------------------------------
# DATASET PATH
# ---------------------------------------------------
DATA_PATH = "breast cancer.csv"

# ---------------------------------------------------
# LOAD AND PREPROCESS DATA
# ---------------------------------------------------
def load_and_preprocess(csv_path):

    # Read CSV
    df = pd.read_csv(csv_path)

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Check diagnosis column
    if "diagnosis" not in df.columns:
        st.error("Column 'diagnosis' not found in dataset.")
        st.stop()

    # Clean diagnosis values
    df["diagnosis"] = (
        df["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Convert labels
    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    # Remove invalid rows
    df = df.dropna(subset=["diagnosis"])

    # Convert target
    y = df["diagnosis"].astype(int)

    # Drop unwanted columns
    drop_cols = ["Unnamed: 32", "id"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Features
    X = df.drop(columns=["diagnosis"])

    # Convert all columns to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill missing values
    X = X.fillna(X.mean())

    feature_names = list(X.columns)

    return X, y, feature_names


# ---------------------------------------------------
# CACHE DATA
# ---------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    return load_and_preprocess(DATA_PATH)


X, y, feature_names = get_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Model Controls")

kernel = st.sidebar.selectbox(
    "Kernel",
    ["linear", "rbf", "poly", "sigmoid"],
    index=1
)

C = st.sidebar.slider(
    "C (Regularization)",
    min_value=0.01,
    max_value=50.0,
    value=1.0,
    step=0.01
)

gamma = st.sidebar.selectbox(
    "Gamma",
    ["scale", "auto"],
    index=0
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

probability = st.sidebar.toggle(
    "Enable Probability",
    value=False
)

random_state = 42

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=probability,
        random_state=random_state
    )

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            digits=4
        )
    }

    return model, scaler, metrics


# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------
with st.spinner("Training model..."):
    model, scaler, metrics = train_model(X, y)

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Accuracy")
    st.metric(
        label="Model Accuracy",
        value=f"{metrics['accuracy']:.4f}"
    )

with col2:
    st.subheader("Confusion Matrix")

    cm = metrics["confusion_matrix"]

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Benign (0)",
            "Actual Malignant (1)"
        ],
        columns=[
            "Predicted Benign (0)",
            "Predicted Malignant (1)"
        ]
    )

    st.dataframe(cm_df, use_container_width=True)

# Classification report
st.subheader("Classification Report")
st.text(metrics["classification_report"])

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------
st.divider()

st.subheader("Predict New Sample")

st.write(
    "Enter values for all features below."
)

inputs = {}

with st.expander("Input Features", expanded=True):

    for name in feature_names:

        median_val = float(X[name].median())

        min_val = float(X[name].min())
        max_val = float(X[name].max())

        inputs[name] = st.number_input(
            label=name,
            value=median_val,
            min_value=min_val,
            max_value=max_val,
            step=0.001
        )

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------
predict_btn = st.button(
    "Predict",
    type="primary"
)

if predict_btn:

    sample_df = pd.DataFrame(
        [[inputs[col] for col in feature_names]],
        columns=feature_names
    )

    # Scale input
    scaled_sample = scaler.transform(sample_df)

    # Prediction
    prediction = model.predict(scaled_sample)[0]

    # Display result
    if prediction == 1:
        st.error("Prediction: Malignant (M)")
    else:
        st.success("Prediction: Benign (B)")

    # Probabilities
    if probability:

        probs = model.predict_proba(scaled_sample)[0]

        st.info(
            f"Benign Probability: {probs[0]:.4f}\n\n"
            f"Malignant Probability: {probs[1]:.4f}"
        )

else:
    st.caption("Enter feature values and click Predict.")