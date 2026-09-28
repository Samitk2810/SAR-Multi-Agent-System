import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "SAML-D.csv"
MODEL_PATH = BASE_DIR / "ml" / "models" / "xgb_sar_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

raw_df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", raw_df.shape)

print(raw_df['Is_laundering'].value_counts())
print(raw_df['Is_laundering'].value_counts(normalize=True))


# ============================================================
# SAMPLING
# Same logic as notebook
# ============================================================

USE_FULL_DATA = False
SAMPLE_NEGATIVE_N = 200_000

if USE_FULL_DATA:
    df = raw_df.copy()
else:
    positives = raw_df[raw_df['Is_laundering'] == 1]
    negatives = raw_df[raw_df['Is_laundering'] == 0].sample(
        n=min(SAMPLE_NEGATIVE_N, (raw_df['Is_laundering'] == 0).sum()),
        random_state=1
    )

    df = pd.concat([positives, negatives]).sample(
        frac=1,
        random_state=1
    ).reset_index(drop=True)

print(f"Rows: {len(df):,}")
print(df['Is_laundering'].value_counts())
print(f"Positive rate: {df['Is_laundering'].mean():.4%}")


# ============================================================
# DATE FEATURES
# Same logic as notebook
# ============================================================

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)


# ============================================================
# SENDER BEHAVIOR
# Same logic as notebook
# ============================================================

sender_stats = df.groupby('Sender_account')['Amount'].agg(
    ['count', 'mean', 'std']
).rename(
    columns={
        'count': 'Sender_txn_count',
        'mean': 'Sender_avg_amount',
        'std': 'Sender_std_amount'
    }
)

df = df.merge(
    sender_stats,
    on='Sender_account',
    how='left'
)

df['Sender_std_amount'] = df['Sender_std_amount'].fillna(0)


# ============================================================
# RECEIVER BEHAVIOR
# Same logic as notebook
# ============================================================

receiver_stats = df.groupby('Receiver_account')['Amount'].agg(
    ['count', 'mean']
).rename(
    columns={
        'count': 'Receiver_txn_count',
        'mean': 'Receiver_avg_amount'
    }
)

df = df.merge(
    receiver_stats,
    on='Receiver_account',
    how='left'
)


# ============================================================
# BEHAVIORAL FEATURES
# Same logic as notebook
# ============================================================

df['Amount_dev_from_sender_avg'] = (
    (df['Amount'] - df['Sender_avg_amount'])
    / (df['Sender_std_amount'] + 1)
)

df['Is_cross_border'] = (
    df['Sender_bank_location'] != df['Receiver_bank_location']
).astype(int)

df['Is_round_amount'] = (
    df['Amount'] % 1000 == 0
).astype(int)

df['Near_reporting_threshold'] = (
    (df['Amount'] >= 9000) &
    (df['Amount'] < 10000)
).astype(int)


# ============================================================
# DROP UNNECESSARY COLUMNS
# Same logic as notebook
# ============================================================

df.drop(
    columns=['Time', 'Date'],
    inplace=True
)


# ============================================================
# MODEL FEATURES
# Same logic as notebook
# ============================================================

leak_and_id_cols = [
    "Is_laundering",
    "Laundering_type",
    "Sender_account",
    "Receiver_account",
]

categorical_cols = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]

X = df.drop(columns=leak_and_id_cols)
y = df["Is_laundering"]


# ============================================================
# CATEGORICAL FEATURES
# Same logic as notebook
# ============================================================

for col in categorical_cols:
    X[col] = X[col].astype('category')


# Downcast numeric dtypes
for col in X.select_dtypes(include=['int64']).columns:
    X[col] = pd.to_numeric(
        X[col],
        downcast='integer'
    )

for col in X.select_dtypes(include=['float64']).columns:
    X[col] = pd.to_numeric(
        X[col],
        downcast='float'
    )


# ============================================================
# TRAIN / TEST SPLIT
# Same logic as notebook
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("Train positive rate:", y_train.mean())
print("Test positive rate:", y_test.mean())


# ============================================================
# XGBOOST
# Same configuration as notebook
# ============================================================

neg, pos = np.bincount(y_train)

base_scale_pos_weight = neg / pos

print(
    f"neg={neg}, pos={pos}, "
    f"base scale_pos_weight={base_scale_pos_weight:.1f}"
)


param_grid = {
    'max_depth': [4, 8],
    'eta': [0.05, 0.2],
    'scale_pos_weight': [
        base_scale_pos_weight,
        base_scale_pos_weight * 1.5
    ],
}


xgb = XGBClassifier(
    eval_metric='aucpr',
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    enable_categorical=True,
)


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='average_precision',
    cv=cv,
    verbose=2,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting XGBoost GridSearch...\n")

grid_search.fit(
    X_train,
    y_train
)

print("\nBest Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_


# ============================================================
# EVALUATION
# Same logic as notebook
# ============================================================

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

avg_precision = average_precision_score(
    y_test,
    test_probabilities
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

print(f"\nTest PR-AUC: {avg_precision:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")


# ============================================================
# THRESHOLD SELECTION
# Same logic as notebook
# ============================================================

precision, recall, pr_thresholds = precision_recall_curve(
    y_test,
    test_probabilities
)

desired_recall = 0.90

valid_idx = np.where(
    recall[:-1] >= desired_recall
)[0]

if len(valid_idx) > 0:

    best_idx = valid_idx[
        np.argmax(precision[valid_idx])
    ]

    chosen_threshold = pr_thresholds[best_idx]

else:

    best_idx = np.argmin(
        np.abs(
            recall[:-1] - desired_recall
        )
    )

    chosen_threshold = pr_thresholds[best_idx]

    print(
        "WARNING: could not reach desired recall; "
        "using closest achievable point."
    )


print(f"\nChosen threshold: {chosen_threshold:.4f}")
print(
    f"Precision at threshold: "
    f"{precision[best_idx]:.4f}"
)
print(
    f"Recall at threshold: "
    f"{recall[best_idx]:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {
    "model": best_model,
    "threshold": chosen_threshold,
    "feature_columns": list(X.columns),
    "categorical_columns": categorical_cols,
    "category_mappings": {
        col: X[col].cat.categories.tolist()
        for col in categorical_cols
    },
}

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        model_package,
        f
    )


print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")
print(MODEL_PATH)