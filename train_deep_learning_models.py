from pathlib import Path
import json
import random

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FOLDER = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
TRAIN_PATH = FOLDER / "train_login_model.csv"
TEST_PATH = FOLDER / "test_login_model.csv"
SEED = 42

LEAKAGE_COLUMNS = {
    "txn_fraud_count",
    "user_prev_fraud_count",
    "txn_count",
    "avg_txn_amount",
    "max_txn_amount",
    "high_risk_txn_rate",
    "merchant_risk_txn_count",
    "beneficiary_mismatch_count",
}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def add_domain_features(frame):
    result = frame.copy()
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
    result["behavior_irregularity"] = result["mouse_velocity"] / (
        result["typing_velocity"] + 1.0
    )
    result["interaction_density"] = (
        result["click_count"] + result["keystroke_count"]
    ) / (result["user_login_count"] + 1.0)
    result["network_travel_interaction"] = (
        result["distance_from_home_km"] * (1 + result["is_proxy"] + result["is_vpn"])
    )
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


class FraudMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def probability(model, values):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(values, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(
        y_true, predictions, output_dict=True, zero_division=0
    )
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
    }


def choose_threshold(y_validation, probabilities, objective):
    rows = []
    for threshold in np.linspace(0.01, 0.99, 197):
        result = evaluate(y_validation, probabilities, threshold)
        rows.append({
            "threshold": threshold,
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "false_positive_count": result["false_positive_count"],
            "false_negative_count": result["false_negative_count"],
        })
    table = pd.DataFrame(rows)
    if objective == "precision":
        selected = table.sort_values(
            ["precision", "f1", "recall"], ascending=False
        ).iloc[0]
    else:
        selected = table.sort_values(
            ["recall", "f1", "precision"], ascending=False
        ).iloc[0]
    return float(selected["threshold"]), table


def train_model(x_train, y_train, positive_weight):
    model = FraudMLP(x_train.shape[1])
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
    )

    model.train()
    for _ in range(100):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
    return model


def main():
    seed_everything()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_columns = [
        c for c in train.columns if c not in LEAKAGE_COLUMNS | {"is_fraud"}
    ]

    x = add_domain_features(train[feature_columns])
    x_test = add_domain_features(test[feature_columns])
    y = train["is_fraud"].astype(int).to_numpy()
    y_test = test["is_fraud"].astype(int).to_numpy()

    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=SEED
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "deep_learning_precision_model": ("precision", 3.0),
        "deep_learning_recall_model": ("recall", 12.0),
    }
    results = {}
    test_probabilities = {}

    for name, (objective, positive_weight) in models.items():
        model = train_model(x_train_scaled, y_train, positive_weight)
        validation_probability = probability(model, x_validation_scaled)
        threshold, threshold_table = choose_threshold(
            y_validation, validation_probability, objective
        )
        test_probability = probability(model, x_test_scaled)
        result = evaluate(y_test, test_probability, threshold)
        result["objective"] = objective
        result["positive_class_weight"] = positive_weight
        result["epochs"] = 100
        result["validation_rows"] = len(x_validation_scaled)
        results[name] = result
        test_probabilities[name] = test_probability

        threshold_table.to_csv(
            FOLDER / f"{name}_threshold_search.csv", index=False
        )
        scored = test.copy()
        scored[f"{name}_probability"] = test_probability
        scored[f"{name}_prediction"] = (
            test_probability >= threshold
        ).astype(int)
        scored.to_csv(FOLDER / f"{name}_scored_test.csv", index=False)
        pd.DataFrame(
            result["confusion_matrix"],
            index=["actual_legitimate", "actual_fraud"],
            columns=["predicted_legitimate", "predicted_fraud"],
        ).to_csv(FOLDER / f"{name}_confusion_matrix.csv")
        torch.save(model.state_dict(), FOLDER / f"{name}.pt")

    comparison = pd.DataFrame([
        {
            "model": name,
            "objective": result["objective"],
            "threshold": result["threshold"],
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "accuracy": result["accuracy"],
            "false_positives": result["false_positive_count"],
            "false_negatives": result["false_negative_count"],
        }
        for name, result in results.items()
    ])
    comparison.to_csv(FOLDER / "deep_learning_model_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    for name, values in test_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, values)
        plt.plot(
            recall,
            precision,
            label=f"{name} AP={results[name]['pr_auc']:.3f}",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Deep-learning precision-recall comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER / "deep_learning_precision_recall_curve.png", dpi=160)
    plt.close()

    summary = {
        "framework": "PyTorch",
        "models": results,
        "feature_count": int(x.shape[1]),
        "engineered_features": [c for c in x.columns if c not in feature_columns],
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
        "test_rows": len(test),
    }
    (FOLDER / "deep_learning_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# Deep-Learning Precision and Recall Models",
        "",
        "A PyTorch multilayer perceptron was trained twice with different positive-class weights and independently selected operating thresholds. The test split was not used for threshold selection.",
        "",
        "## Comparison",
        "",
        comparison.to_string(index=False),
        "",
    ]
    for name, result in results.items():
        report.extend([
            f"## {name}",
            "",
            f"- Objective: {result['objective']}",
            f"- Positive-class weight: {result['positive_class_weight']}",
            f"- Threshold: {result['threshold']:.3f}",
            f"- Precision: {result['precision']:.4f}",
            f"- Recall: {result['recall']:.4f}",
            f"- F1: {result['f1']:.4f}",
            f"- PR-AUC: {result['pr_auc']:.4f}",
            f"- False positives: {result['false_positive_count']}",
            f"- False negatives: {result['false_negative_count']}",
            "",
            pd.DataFrame(
                result["confusion_matrix"],
                index=["actual_legitimate", "actual_fraud"],
                columns=["predicted_legitimate", "predicted_fraud"],
            ).to_string(),
            "",
        ])
    report.extend([
        "## Usage",
        "",
        "Use the precision model for high-confidence automated actions and the recall model for broad investigation queues where missed fraud is more costly than analyst review volume.",
    ])
    (FOLDER / "deep_learning_models_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print("Created separate deep-learning precision and recall models.")


if __name__ == "__main__":
    main()
