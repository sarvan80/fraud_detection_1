from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from renewed_modeling_utils import BASE_DIR, EXTENDED_LEAKAGE_COLUMNS, SEED


USERS_PATH = BASE_DIR / "users_v2.csv"
LOGINS_PATH = BASE_DIR / "login_events_v2.csv"
TRANSACTIONS_PATH = BASE_DIR / "transactions_v2.csv"


def prev_expanding_mean(series: pd.Series) -> np.ndarray:
    return series.shift(1).expanding().mean().bfill().to_numpy()


def main() -> None:
    users = pd.read_csv(USERS_PATH)
    logins = pd.read_csv(LOGINS_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)

    for frame, column in [(users, "signup_date"), (logins, "event_time"), (transactions, "txn_time")]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    logins = logins.merge(users, on="user_id", how="left", suffixes=("", "_user"))

    transaction_direct = transactions.rename(columns={
        "txn_time": "txn_event_time",
        "mfa_required": "txn_mfa_required",
        "mfa_approved": "txn_mfa_approved",
        "merchant_country": "txn_merchant_country",
        "payment_method": "txn_payment_method",
    })
    direct_drop = {"device_id", "device_fingerprint", "ip_address", "is_fraud", "fraud_type"}
    transaction_direct = transaction_direct[[c for c in transaction_direct.columns if c not in direct_drop]]
    logins = logins.merge(transaction_direct, on="user_id", how="left")

    user_txn = transactions.groupby("user_id").agg(
        txn_count=("txn_id", "count"),
        avg_txn_amount=("txn_amount", "mean"),
        max_txn_amount=("txn_amount", "max"),
        txn_fraud_count=("is_fraud", "sum"),
        high_risk_txn_rate=("txn_risk_score", lambda values: (values > 80).mean()),
        merchant_risk_txn_count=("merchant_risk_flag", "sum"),
        beneficiary_mismatch_count=("beneficiary_country_mismatch", "sum"),
    ).reset_index()
    logins = logins.merge(user_txn, on="user_id", how="left")

    logins = logins.sort_values(["user_id", "event_time"]).reset_index(drop=True)
    logins["user_login_count"] = logins.groupby("user_id").cumcount() + 1
    logins["user_prev_fraud_count"] = logins.groupby("user_id")["is_fraud"].transform(lambda series: series.shift(1).fillna(0).cumsum())
    logins["user_prev_risk_mean"] = logins.groupby("user_id")["login_risk_score"].transform(prev_expanding_mean)
    logins["user_prev_failed_mean"] = logins.groupby("user_id")["failed_login_count_24h"].transform(prev_expanding_mean)
    logins["user_prev_new_device_rate"] = logins.groupby("user_id")["new_device_flag"].transform(prev_expanding_mean)
    logins["user_prev_proxy_rate"] = logins.groupby("user_id")["is_proxy"].transform(prev_expanding_mean)

    logins["geo_distance_km"] = np.sqrt((logins["ip_lat"] - logins["home_lat"]) ** 2 + (logins["ip_lon"] - logins["home_lon"]) ** 2) * 111.32
    logins["distance_from_home_km"] = np.round(logins["geo_distance_km"], 2)
    logins["velocity_flag"] = (logins["distance_from_home_km"] > 2000).astype(int)
    logins["travel_risk_score"] = (
        5 * logins["impossible_travel_flag"]
        + 3 * logins["is_proxy"]
        + 2 * logins["is_vpn"]
        + 2 * logins["new_device_flag"]
        + 2 * logins["velocity_flag"]
    )

    fill_zero_columns = [
        "txn_count",
        "avg_txn_amount",
        "max_txn_amount",
        "txn_fraud_count",
        "high_risk_txn_rate",
        "merchant_risk_txn_count",
        "beneficiary_mismatch_count",
        "user_prev_risk_mean",
        "user_prev_failed_mean",
        "user_prev_new_device_rate",
        "user_prev_proxy_rate",
    ]
    for column in fill_zero_columns:
        logins[column] = logins[column].fillna(0)

    boolean_like_columns = [column for column in logins.columns if logins[column].dropna().isin([0, 1, True, False]).all()]
    for column in boolean_like_columns:
        logins[column] = logins[column].fillna(0).astype(int)

    drop_from_model = {
        "login_id",
        "user_id",
        "txn_id",
        "event_time",
        "txn_event_time",
        "signup_date",
        "fraud_type",
        "fraud_score",
        "login_result",
        "device_id",
        "device_fingerprint",
        "ip_address",
        "ja3_hash",
        "tcp_stack_fingerprint",
        "asn_for_ip",
        "home_city",
        "home_city_user",
        "country_user",
        "geo_distance_km",
    }

    categorical_columns = [
        column
        for column in logins.select_dtypes(include=["object", "category"]).columns
        if column not in drop_from_model
    ]
    model_df = pd.get_dummies(logins, columns=categorical_columns, prefix=categorical_columns, drop_first=False)

    columns_to_keep = [column for column in model_df.columns if column not in drop_from_model]
    model_df = model_df[columns_to_keep]
    bool_columns = model_df.select_dtypes(include=["bool"]).columns
    if len(bool_columns) > 0:
        model_df[bool_columns] = model_df[bool_columns].astype(int)
    for column in model_df.columns:
        if column != "is_fraud":
            model_df[column] = model_df[column].fillna(0)
    model_df["is_fraud"] = model_df["is_fraud"].astype(int)

    all_nan_columns = model_df.columns[model_df.isna().all()].tolist()
    if all_nan_columns:
        raise ValueError(f"Unexpected all-NaN columns: {all_nan_columns}")

    X = model_df.drop(columns=["is_fraud"])
    y = model_df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    train_final = X_train.copy()
    train_final["is_fraud"] = y_train.values
    test_final = X_test.copy()
    test_final["is_fraud"] = y_test.values

    model_df.to_csv(BASE_DIR / "model_ready_login_v2.csv", index=False)
    train_final.to_csv(BASE_DIR / "train_login_model_v2.csv", index=False)
    test_final.to_csv(BASE_DIR / "test_login_model_v2.csv", index=False)

    original_users = pd.read_csv(Path(__file__).resolve().parents[1] / "users_10000.csv", nrows=1)
    original_logins = pd.read_csv(Path(__file__).resolve().parents[1] / "login_events_10000.csv", nrows=1)
    original_transactions = pd.read_csv(Path(__file__).resolve().parents[1] / "transactions_10000.csv", nrows=1)
    summary = {
        "rows": {
            "users_v2": int(len(users)),
            "login_events_v2": int(len(logins)),
            "transactions_v2": int(len(transactions)),
            "model_ready_login_v2": int(len(model_df)),
            "train_login_model_v2": int(len(train_final)),
            "test_login_model_v2": int(len(test_final)),
        },
        "fraud_rate": {
            "overall": float(y.mean()),
            "train": float(y_train.mean()),
            "test": float(y_test.mean()),
        },
        "feature_counts": {
            "original_users": int(len(original_users.columns)),
            "renewed_users": int(len(users.columns)),
            "original_logins": int(len(original_logins.columns)),
            "renewed_logins": int(len(pd.read_csv(LOGINS_PATH, nrows=1).columns)),
            "original_transactions": int(len(original_transactions.columns)),
            "renewed_transactions": int(len(pd.read_csv(TRANSACTIONS_PATH, nrows=1).columns)),
            "model_ready_features": int(X.shape[1]),
        },
        "excluded_identifiers_or_post_outcomes": sorted(drop_from_model),
        "documented_leakage_columns": sorted(EXTENDED_LEAKAGE_COLUMNS),
        "categorical_columns_one_hot_encoded": sorted(categorical_columns),
        "all_nan_columns": all_nan_columns,
    }
    (BASE_DIR / "dataset_build_summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved model_ready_login_v2.csv")
    print("Saved train_login_model_v2.csv")
    print("Saved test_login_model_v2.csv")
    print(f"Overall fraud rate: {y.mean():.4f}")
    print(f"Model-ready feature count: {X.shape[1]}")


if __name__ == "__main__":
    main()
