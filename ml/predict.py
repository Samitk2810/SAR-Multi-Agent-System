import pickle
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "new_transactions.csv"
MODEL_PATH = BASE_DIR / "ml" / "models" / "xgb_sar_model.pkl"
OUTPUT_PATH = BASE_DIR / "data" / "screening_results.csv"


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

with open(MODEL_PATH, "rb") as f:
    package = pickle.load(f)

model = package["model"]
threshold = 0.87
feature_columns = package["feature_columns"]
categorical_columns = package["categorical_columns"]

print(f"Model loaded: {MODEL_PATH}")
print(f"Threshold: {threshold:.4f}")


# ============================================================
# LOAD NEW RAW TRANSACTIONS
# ============================================================

df = pd.read_csv(INPUT_PATH)

print(f"\nInput transactions: {len(df)}")


# ============================================================
# REMOVE ROWS WITH MISSING VALUES
# ============================================================

before = len(df)

df = df.dropna().reset_index(drop=True)

removed = before - len(df)

print(f"Rows removed because of missing values: {removed}")
print(f"Rows remaining: {len(df)}")


# ============================================================
# DATE FEATURES
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)


# ============================================================
# SENDER BEHAVIOR
# ============================================================

sender_stats = df.groupby("Sender_account")["Amount"].agg(
    ["count", "mean", "std"]
).rename(
    columns={
        "count": "Sender_txn_count",
        "mean": "Sender_avg_amount",
        "std": "Sender_std_amount"
    }
)

df = df.merge(
    sender_stats,
    on="Sender_account",
    how="left"
)

df["Sender_std_amount"] = df["Sender_std_amount"].fillna(0)


# ============================================================
# RECEIVER BEHAVIOR
# ============================================================

receiver_stats = df.groupby("Receiver_account")["Amount"].agg(
    ["count", "mean"]
).rename(
    columns={
        "count": "Receiver_txn_count",
        "mean": "Receiver_avg_amount"
    }
)

df = df.merge(
    receiver_stats,
    on="Receiver_account",
    how="left"
)


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

df["Amount_dev_from_sender_avg"] = (
    (df["Amount"] - df["Sender_avg_amount"])
    / (df["Sender_std_amount"] + 1)
)

df["Is_cross_border"] = (
    df["Sender_bank_location"]
    != df["Receiver_bank_location"]
).astype(int)

df["Is_round_amount"] = (
    df["Amount"] % 1000 == 0
).astype(int)

df["Near_reporting_threshold"] = (
    (df["Amount"] >= 9000)
    & (df["Amount"] < 10000)
).astype(int)


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

X = df.drop(
    columns=[
        "Sender_account",
        "Receiver_account",
        "Time", "Date"
    ]
)


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

category_mappings = package["category_mappings"]

for col in categorical_columns:
    X[col] = pd.Categorical(
        X[col],
        categories=category_mappings[col]
    )


# ============================================================
# SAME FEATURE ORDER AS TRAINING
# ============================================================

X = X[feature_columns]


# ============================================================
# XGBOOST PREDICTION
# ============================================================

probabilities = model.predict_proba(X)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


# ============================================================
# ADD RESULTS
# ============================================================

df["risk_probability"] = probabilities
df["sar_worthy"] = predictions


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("XGBOOST SCREENING COMPLETE")
print("========================================")

for index, row in df.iterrows():

    status = (
        "SAR-WORTHY"
        if row["sar_worthy"] == 1
        else "NOT SAR-WORTHY"
    )

    print(
        f"{index + 1}. "
        f"{status} | "
        f"Amount: {row['Amount']} | "
        f"Risk: {row['risk_probability']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_PATH)