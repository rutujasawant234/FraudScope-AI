
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="FraudScope AI", page_icon="🔍", layout="wide")

st.title("🔍 FraudScope AI")
st.subheader("Intelligent Financial Fraud Detection System")
st.caption("Educational demonstration — not a real banking fraud detector.")

# 1. Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("transactions.csv")
        st.sidebar.info("Using transactions.csv")
    except FileNotFoundError:
        st.error("Upload a CSV file to begin.")
        st.stop()

# Data Preprocessing
st.header("Data Preprocessing")

original_rows = len(df)

# Remove duplicate transactions
df = df.drop_duplicates()

# Convert required numeric columns
numeric_columns = ["amount", "hour", "transactions_last_hour", "label"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing or invalid numeric values
df = df.dropna(subset=numeric_columns + ["transaction_id"])

# Validate transaction values
df = df[
    (df["amount"] >= 0) &
    (df["hour"].between(0, 23)) &
    (df["transactions_last_hour"] >= 0) &
    (df["label"].isin([0, 1]))
].copy()

st.write("Rows before preprocessing:", original_rows)
st.write("Rows after preprocessing:", len(df))
st.write("Duplicate and invalid rows removed:", original_rows - len(df))


# Feature Engineering
st.header("Feature Engineering")

# Identify night transactions (12 AM to 5 AM)
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

# Identify unusually high transaction amounts
amount_threshold = df["amount"].quantile(0.75)
df["high_amount"] = (df["amount"] > amount_threshold).astype(int)

# Identify high transaction frequency
frequency_threshold = df["transactions_last_hour"].quantile(0.75)
df["high_frequency"] = (
    df["transactions_last_hour"] > frequency_threshold
).astype(int)

# Use logarithm to reduce the influence of large amounts
df["log_amount"] = np.log1p(df["amount"])

st.write("New features created successfully:")
st.dataframe(
    df[[
        "transaction_id", "amount", "hour",
        "transactions_last_hour", "is_night",
        "high_amount", "high_frequency", "log_amount"
    ]],
    use_container_width=True
)

features = [
    "amount", "hour", "transactions_last_hour",
    "is_night", "high_amount", "high_frequency", "log_amount"
]
required = ["transaction_id"] + features + ["label"]

if not all(c in df.columns for c in required):
    st.error("CSV must contain: transaction_id, amount, hour, transactions_last_hour, label.")
    st.stop()

if df.empty or df[required].isnull().any().any():
    st.error("The dataset is empty or has missing required values.")
    st.stop()

if not set(df["label"].unique()).issubset({0, 1}) or df["label"].nunique() != 2:
    st.error("The label column must contain both 0 (legitimate) and 1 (fraud).")
    st.stop()

if len(df) < 10:
    st.warning("Very small dataset. Model metrics will be unreliable.")

# Validate feature values
try:
    df[features] = df[features].apply(pd.to_numeric, errors="raise")
except (ValueError, TypeError):
    st.error("The feature columns must contain numeric values.")
    st.stop()

if not np.isfinite(df[features].to_numpy(dtype=float)).all():
    st.error("Feature columns must contain finite numeric values.")
    st.stop()

# 2. Holdout evaluation
st.header("1. Model Evaluation")

if len(df) >= 10 and df["label"].value_counts().min() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["label"],
        test_size=0.3, random_state=42, stratify=df["label"]
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2%}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2%}")
    m4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.2%}")

    st.write("Confusion Matrix (test data)")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(cm,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"])
    )
else:
    st.warning("Not enough examples per class for a reliable holdout evaluation.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features], df["label"])

# 3. Analyze transactions using full dataset model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df[features], df["label"])

results = df.copy()
results["Fraud Probability (%)"] = (
    model.predict_proba(results[features])[:, 1] * 100
).round(2)
results["Predicted Label"] = model.predict(results[features])
results["Prediction"] = results["Predicted Label"].map({
    0: "Legitimate", 1: "Fraud"
})

# Risk categories are screening thresholds, not calibrated probabilities
results["Risk Level"] = pd.cut(
    results["Fraud Probability (%)"],
    bins=[-0.01, 30, 70, 100],
    labels=["Low", "Medium", "High"]
).astype(str)

# 4. Isolation Forest

# 4. Isolation Forest
st.header("2. Anomaly Detection")

iso_scaler = StandardScaler()
iso_X = iso_scaler.fit_transform(results[features])

detector = IsolationForest(random_state=42, contamination="auto")
results["Pattern"] = detector.fit_predict(iso_X)
results["Pattern"] = results["Pattern"].map({
    1: "Normal", -1: "Unusual"
})
results["Anomaly Score"] = detector.decision_function(iso_X).round(4)

st.subheader("Isolation Forest Output")
st.dataframe(
    results[["transaction_id", "amount", "Pattern", "Anomaly Score"]],
    use_container_width=True
)

# 5. DBSCAN clustering

# 5. DBSCAN clustering
st.header("3. DBSCAN Clustering")

cluster_model = DBSCAN(eps=1.5, min_samples=2)
results["Cluster"] = cluster_model.fit_predict(iso_X)

results["Cluster Status"] = results["Cluster"].apply(
    lambda x: "Noise Point" if x == -1 else f"Cluster {x}"
)

st.subheader("DBSCAN Output")
st.dataframe(
    results[["transaction_id", "amount", "Cluster", "Cluster Status"]],
    use_container_width=True
)


# Dashboard metrics
st.header("4. Dashboard Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", len(results))
c2.metric("Fraud Labels", int((results["label"] == 1).sum()))
c3.metric("Unusual Patterns", int((results["Pattern"] == "Unusual").sum()))
c4.metric("DBSCAN Noise", int((results["Cluster"] == -1).sum()))

st.subheader("Transaction Amounts")
st.bar_chart(results.set_index("transaction_id")["amount"])

st.subheader("Fraud Probability (%)")
st.bar_chart(results.set_index("transaction_id")["Fraud Probability (%)"])

# 6. Interactive filters
st.header("5. Interactive Transaction Filters")
min_amount = float(results["amount"].min())
max_amount = float(results["amount"].max())

if min_amount < max_amount:
    amount_range = st.slider(
        "Transaction amount range",
        min_value=min_amount, max_value=max_amount,
        value=(min_amount, max_amount)
    )
else:
    amount_range = (min_amount, max_amount)

risk_choice = st.selectbox(
    "Risk level", ["All", "Low", "Medium", "High"]
)
pattern_choice = st.selectbox(
    "Pattern", ["All", "Normal", "Unusual"]
)

filtered = results[
    results["amount"].between(amount_range[0], amount_range[1])
].copy()

if risk_choice != "All":
    filtered = filtered[filtered["Risk Level"] == risk_choice]
if pattern_choice != "All":
    filtered = filtered[filtered["Pattern"] == pattern_choice]

st.subheader("Analysis Results")
st.dataframe(filtered, use_container_width=True)

st.download_button(
    "Download Analysis CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="fraudscope_results.csv",
    mime="text/csv"
)

# 7. Apriori association rules
st.header("6. Apriori Association Rule Mining")

items = pd.DataFrame({
    "High Amount": results["amount"] >= results["amount"].quantile(0.75),
    "Night Transaction": results["hour"].isin([0, 1, 2, 3, 4, 5]),
    "Many Recent Transactions": results["transactions_last_hour"] >=
        results["transactions_last_hour"].quantile(0.75),
    "Fraud Label": results["label"] == 1
})

if len(items) >= 2:
    try:
        frequent = apriori(items, min_support=0.1, use_colnames=True)

        if len(frequent) >= 2:
            rules = association_rules(
                frequent, metric="confidence", min_threshold=0.5
            )
            if not rules.empty:
                rules["antecedents"] = rules["antecedents"].apply(
                    lambda x: ", ".join(sorted(x))
                )
                rules["consequents"] = rules["consequents"].apply(
                    lambda x: ", ".join(sorted(x))
                )
                st.dataframe(rules, use_container_width=True)
            else:
                st.info("No association rules met the current threshold.")
        else:
            st.info("Not enough frequent itemsets were found.")
    except (ValueError, KeyError) as e:
        st.info(f"Association rules could not be generated: {e}")

st.warning(
    "Synthetic/small datasets can produce misleading metrics and rules. "
    "Fraud probabilities are model scores, not necessarily calibrated risks. "
    "Anomalies and DBSCAN noise points are not confirmed fraud."
)

