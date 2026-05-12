import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


st.set_page_config(page_title="SVC Breast Cancer App", layout="wide")

st.title("Breast Cancer Classification (SVC)")
st.write("This app trains an SVC model using `beast cancer.csv` and lets you predict for a new sample.")

DATA_PATH = "beast cancer.csv"


def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)

    # Basic sanity checks
    if "diagnosis" not in df.columns:
        raise ValueError("Expected column 'diagnosis' in the dataset.")

    # Map diagnosis to 0/1 if needed
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

    # Drop unwanted columns if they exist
    drop_cols = ["Unnamed: 32", "id"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Ensure label present and clean
    df = df.dropna(subset=["diagnosis"]).copy()

    # Separate X/y
    y = df["diagnosis"].astype(int)
    X = df.drop(columns=["diagnosis"])

    # Fill missing numeric values
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True))

    feature_names = list(X.columns)
    return X, y, feature_names


@st.cache_data(show_spinner=False)
def get_data():
    X, y, feature_names = load_and_preprocess(DATA_PATH)
    return X, y, feature_names


X, y, feature_names = get_data()

# Sidebar controls
st.sidebar.header("SVC Controls")

kernel = "rbf"  # match svc1.ipynb

# Match svc1.ipynb random_state
random_state = 42

# Hyperparameters (optional UI)
# svc1.ipynb did not specify C/gamma explicitly; default for SVC is C=1.0, gamma='scale'
# but we let user tune while keeping kernel + random_state consistent.
probability = st.sidebar.toggle("Enable probability estimates", value=False)

C = st.sidebar.slider("C (regularization)", min_value=0.01, max_value=50.0, value=1.0, step=0.01)
gamma = st.sidebar.selectbox("gamma", options=["scale", "auto"], index=0)

# Train/test split
st.sidebar.subheader("Train/Test")
test_size = st.sidebar.slider("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(
        kernel=kernel,
        C=float(C),
        gamma=gamma,
        probability=bool(probability),
        random_state=random_state,
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }

    return model, scaler, metrics


with st.spinner("Training model..."):
    model, scaler, metrics = train_model(X, y)


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")

with col2:
    st.subheader("Confusion Matrix")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual: Benign (0)", "Actual: Malignant (1)"], columns=["Pred: Benign (0)", "Pred: Malignant (1)"])
    st.dataframe(cm_df, use_container_width=True)

st.subheader("Classification Report")
st.text(metrics["classification_report"])


st.divider()

st.subheader("Predict for a New Sample")
st.write("Enter values for all features. The app will scale them using the same `StandardScaler` learned during training.")

inputs = {}

# Create inputs in a stable order
with st.expander("Input features", expanded=True):
    for name in feature_names:
        # make sure inputs dict is updated for every feature

        # reasonable default range from dataset values is unknown; use broad min/max
        # Also keep step small enough for user adjustment.
        val = float(X[name].median())
        min_val = float(X[name].quantile(0.01))
        max_val = float(X[name].quantile(0.99))

        # Expand bounds a bit to reduce clipping feel
        span = (max_val - min_val) if max_val > min_val else 1.0
        min_val -= 0.1 * span
        max_val += 0.1 * span

        inputs[name] = st.number_input(
            name,
            value=val,
            min_value=min_val,
            max_value=max_val,
            step=0.001,
        )


predict_btn = st.button("Predict", type="primary")

if predict_btn:
    st.write("Using current input values...")

    # Create dataframe to preserve feature names for sklearn warnings/debugging
    sample_df = pd.DataFrame([[inputs[n] for n in feature_names]], columns=feature_names)
    st.write("First 5 feature values:")
    st.dataframe(sample_df.iloc[:, :5], use_container_width=True)

    scaled_sample = scaler.transform(sample_df)

    pred = model.predict(scaled_sample)[0]

    label = "Malignant (M)" if pred == 1 else "Benign (B)"

    st.success(f"Prediction: {label}")

    if probability:
        proba = model.predict_proba(scaled_sample)[0]
        p_benign = float(proba[0])
        p_malignant = float(proba[1])
        st.info(f"Probability — Benign: {p_benign:.4f} | Malignant: {p_malignant:.4f}")
    else:
        st.caption("Probability estimates are disabled. Enable them in the sidebar to display class probabilities.")
else:
    st.caption("Set the feature values above, then click Predict.")


