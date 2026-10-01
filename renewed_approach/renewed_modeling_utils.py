from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train_login_model_v2.csv"
TEST_PATH = BASE_DIR / "test_login_model_v2.csv"
MODEL_READY_PATH = BASE_DIR / "model_ready_login_v2.csv"
TARGET = "is_fraud"
SEED = 42

BASE_LEAKAGE_COLUMNS = {
    "txn_count",
    "avg_txn_amount",
    "max_txn_amount",
    "txn_fraud_count",
    "high_risk_txn_rate",
    "merchant_risk_txn_count",
    "beneficiary_mismatch_count",
    "user_prev_fraud_count",
}

EXTENDED_LEAKAGE_COLUMNS = BASE_LEAKAGE_COLUMNS | {
    "shared_ip_fraud_density_30d",
    "ip_account_takeover_cluster_flag",
    "ip_reuse_attack_score",
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    required_defaults = {
        "new_device_flag": 0,
        "is_proxy": 0,
        "is_vpn": 0,
        "impossible_travel_flag": 0,
        "mfa_approved": 0,
        "failed_login_count_24h": 0,
        "session_risk_score": 0,
        "device_risk_score": 0,
        "login_risk_score": 0,
        "user_prev_risk_mean": 0,
        "mouse_velocity": 0.0,
        "typing_velocity": 0.0,
        "click_count": 0,
        "keystroke_count": 0,
        "user_login_count": 1,
        "distance_from_home_km": 0.0,
        "device_novelty_score": 0.0,
        "fingerprint_population_rarity": 0.0,
        "fingerprint_similarity_to_fraud_devices": 0.0,
        "device_spoofed": 0,
        "suspicious_fingerprint": 0,
        "device_network_novelty_combo": 0,
        "device_identity_confidence": 50.0,
        "cumulative_identity_risk_index": 0.0,
        "identity_event_cluster_score": 0.0,
        "password_reset_freq_30d": 0.0,
        "mfa_prompt_count_1h": 0.0,
        "failed_logins_before_reset_24h": 0.0,
        "breach_exposure_recency_flag": 0,
        "identity_change_on_new_device_combo": 0,
        "ato_sequence_score": 0.0,
        "network_risk_score": 0.0,
        "geo_velocity_risk_score": 0.0,
        "request_burst_rate_5m": 0.0,
        "tls_cipher_anomaly_score": 0.0,
        "network_fingerprint_mismatch_flag": 0,
        "anonymized_ip_persistence_7d": 0.0,
        "bot_likelihood_score": 0.0,
        "interaction_event_count_pre_submit": 0.0,
        "mouse_movement_entropy": 0.0,
        "time_to_login_submit_ms": 1.0,
        "session_journey_anomaly": 0.0,
        "new_payee_to_transfer_mins": 720.0,
        "first_time_high_value_transfer_after_profile_change": 0,
        "txn_amount": 0.0,
        "txn_risk_score": 0.0,
    }
    for column, default in required_defaults.items():
        if column not in result.columns:
            result[column] = default

    result["risk_signal_count"] = (
        result["new_device_flag"]
        + result["is_proxy"]
        + result["is_vpn"]
        + result["impossible_travel_flag"]
        + (result["mfa_approved"] == 0).astype(int)
    )
    result["authentication_pressure"] = (
        result["failed_login_count_24h"] * 2
        + result["session_risk_score"]
        + result["device_risk_score"]
    )
    result["risk_score_gap"] = result["login_risk_score"] - result["user_prev_risk_mean"]
    result["behavior_irregularity"] = result["mouse_velocity"] / (result["typing_velocity"] + 1.0)
    result["interaction_density"] = (
        result["click_count"] + result["keystroke_count"]
    ) / (result["user_login_count"] + 1.0)
    result["network_travel_interaction"] = (
        result["distance_from_home_km"] * (1 + result["is_proxy"] + result["is_vpn"])
    )
    result["device_anomaly_index"] = (
        0.35 * result["device_novelty_score"]
        + 0.20 * result["fingerprint_population_rarity"]
        + 0.15 * result["fingerprint_similarity_to_fraud_devices"]
        + 9.0 * result["device_spoofed"]
        + 9.0 * result["suspicious_fingerprint"]
        + 7.0 * result["device_network_novelty_combo"]
    )
    result["identity_takeover_pressure"] = (
        0.45 * result["cumulative_identity_risk_index"]
        + 8.0 * result["identity_event_cluster_score"]
        + 6.0 * result["identity_change_on_new_device_combo"]
        + 2.5 * result["password_reset_freq_30d"]
        + 2.0 * result["mfa_prompt_count_1h"]
        + 1.3 * result["failed_logins_before_reset_24h"]
        + 10.0 * result["breach_exposure_recency_flag"]
    )
    result["network_evasion_index"] = (
        0.40 * result["network_risk_score"]
        + 0.25 * result["geo_velocity_risk_score"]
        + 0.20 * result["tls_cipher_anomaly_score"]
        + 0.12 * result["request_burst_rate_5m"]
        + 6.0 * result["network_fingerprint_mismatch_flag"]
        + 2.5 * result["anonymized_ip_persistence_7d"]
    )
    result["bot_behavior_index"] = (
        0.45 * result["bot_likelihood_score"]
        + 0.15 * (100 - result["mouse_movement_entropy"] * 100)
        + 0.15 * (1000 / (result["time_to_login_submit_ms"] + 10.0))
        + 0.10 * result["request_burst_rate_5m"]
        + 0.15 * result["session_journey_anomaly"]
    )
    result["cross_surface_takeover_score"] = (
        5.0 * result["device_network_novelty_combo"]
        + 4.0 * result["identity_change_on_new_device_combo"]
        + 8.0 * result["first_time_high_value_transfer_after_profile_change"]
        + 4.0 * result["ato_sequence_score"]
        + 250.0 / (result["new_payee_to_transfer_mins"] + 5.0)
    )
    result["trust_gap"] = (
        (100 - result["device_identity_confidence"])
        + 0.55 * result["network_risk_score"]
        + 0.35 * result["cumulative_identity_risk_index"]
        + 0.20 * result["txn_risk_score"]
    )
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def evaluate_predictions(y_true: Iterable[int], probabilities: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "false_positive_count": int(((y_true == 0) & (predictions == 1)).sum()),
        "false_negative_count": int(((y_true == 1) & (predictions == 0)).sum()),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": report,
    }


def threshold_search(
    y_true: Iterable[int],
    probabilities: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true, probabilities, float(threshold))
        rows.append({
            "threshold": float(threshold),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "false_positive_count": metrics["false_positive_count"],
            "false_negative_count": metrics["false_negative_count"],
        })
    return pd.DataFrame(rows)


def save_confusion_matrix_artifacts(matrix: list[list[int]], csv_path: Path, png_path: Path | None, title: str) -> None:
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_legitimate", "actual_fraud"],
        columns=["predicted_legitimate", "predicted_fraud"],
    )
    matrix_df.to_csv(csv_path)
    if png_path is not None:
        plt.figure(figsize=(5, 4))
        plt.imshow(matrix_df.values, cmap="Blues")
        plt.title(title)
        plt.xticks([0, 1], matrix_df.columns, rotation=15)
        plt.yticks([0, 1], matrix_df.index)
        for row in range(2):
            for column in range(2):
                plt.text(column, row, int(matrix_df.iloc[row, column]), ha="center", va="center", color="#0f172a")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()


def save_feature_importance(importances: np.ndarray, feature_names: list[str], csv_path: Path, png_path: Path, title: str, top_n: int = 20) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    importance.to_csv(csv_path, index=False)

    top = importance.head(top_n).sort_values("importance")
    plt.figure(figsize=(10, max(6, top_n * 0.28)))
    plt.barh(top["feature"], top["importance"], color="#2563eb")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    return importance


def save_pr_curve(curves: list[tuple[str, np.ndarray, np.ndarray]], png_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for label, y_true, probabilities in curves:
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        ap = average_precision_score(y_true, probabilities)
        plt.plot(recall, precision, label=f"{label} AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    def fmt(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator] + body)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
