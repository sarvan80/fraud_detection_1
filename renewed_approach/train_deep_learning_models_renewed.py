from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from renewed_modeling_utils import (
    BASE_DIR,
    EXTENDED_LEAKAGE_COLUMNS,
    SEED,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    add_domain_features,
    evaluate_predictions,
    markdown_table,
    save_confusion_matrix_artifacts,
    save_pr_curve,
    threshold_search,
    write_json,
)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class FraudMLP(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        hidden_one = max(128, min(256, input_size))
        hidden_two = max(64, hidden_one // 2)
        hidden_three = max(24, hidden_two // 2)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_one),
            nn.BatchNorm1d(hidden_one),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_one, hidden_two),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_two, hidden_three),
            nn.ReLU(),
            nn.Linear(hidden_three, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


def probability(model: FraudMLP, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(values, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def choose_threshold(y_validation: np.ndarray, probabilities: np.ndarray, objective: str) -> tuple[float, pd.DataFrame]:
    table = threshold_search(y_validation, probabilities, np.linspace(0.01, 0.99, 197))
    if objective == "precision":
        selected = table.sort_values(["precision", "f1", "recall"], ascending=False).iloc[0]
    else:
        selected = table.sort_values(["recall", "f1", "precision"], ascending=False).iloc[0]
    return float(selected["threshold"]), table


def train_model(x_train: np.ndarray, y_train: np.ndarray, positive_weight: float) -> FraudMLP:
    model = FraudMLP(x_train.shape[1])
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], dtype=torch.float32))

    model.train()
    for _ in range(100):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
    return model


def main() -> None:
    seed_everything()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_columns = [column for column in train.columns if column not in EXTENDED_LEAKAGE_COLUMNS | {TARGET}]

    X_full = add_domain_features(train[feature_columns])
    X_test = add_domain_features(test[feature_columns])
    y_full = train[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_full,
        y_full,
        test_size=0.25,
        stratify=y_full,
        random_state=SEED,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "deep_learning_precision_model_renewed": ("precision", 3.0),
        "deep_learning_recall_model_renewed": ("recall", 12.0),
    }

    results = {}
    test_probabilities = {}

    for name, (objective, positive_weight) in models.items():
        model = train_model(X_train_scaled, y_train, positive_weight)
        validation_probability = probability(model, X_validation_scaled)
        threshold, threshold_table = choose_threshold(y_validation, validation_probability, objective)
        test_probability = probability(model, X_test_scaled)

        result = evaluate_predictions(y_test, test_probability, threshold)
        result["objective"] = objective
        result["positive_class_weight"] = positive_weight
        result["epochs"] = 100
        result["validation_rows"] = len(X_validation_scaled)
        results[name] = result
        test_probabilities[name] = test_probability

        threshold_table.to_csv(BASE_DIR / f"{name}_threshold_search.csv", index=False)
        scored = test.copy()
        scored[f"{name}_probability"] = test_probability
        scored[f"{name}_prediction"] = (test_probability >= threshold).astype(int)
        scored.to_csv(BASE_DIR / f"{name}_scored_test.csv", index=False)
        save_confusion_matrix_artifacts(
            result["confusion_matrix"],
            BASE_DIR / f"{name}_confusion_matrix.csv",
            BASE_DIR / f"{name}_confusion_matrix.png",
            f"{name} confusion matrix",
        )
        torch.save(model.state_dict(), BASE_DIR / f"{name}.pt")

    comparison_rows = [
        {
            "model": name,
            "objective": result["objective"],
            "threshold": result["threshold"],
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        }
        for name, result in results.items()
    ]
    pd.DataFrame(comparison_rows).to_csv(BASE_DIR / "deep_learning_model_comparison_renewed.csv", index=False)
    save_pr_curve(
        [(name, y_test, probability_values) for name, probability_values in test_probabilities.items()],
        BASE_DIR / "deep_learning_precision_recall_curve_renewed.png",
        "Renewed deep-learning model comparison",
    )

    summary = {
        "framework": "PyTorch",
        "models": results,
        "feature_count": int(X_full.shape[1]),
        "engineered_features": [column for column in X_full.columns if column not in feature_columns],
        "excluded_leakage_columns": sorted(EXTENDED_LEAKAGE_COLUMNS),
        "test_rows": len(test),
    }
    write_json(BASE_DIR / "deep_learning_metrics_renewed.json", summary)

    report_lines = [
        "# Renewed Deep-Learning Precision and Recall Models",
        "",
        "The renewed PyTorch MLP scales its input layer to the larger enriched feature space and keeps the original precision-optimized and recall-optimized training pattern.",
        "",
        "## Model comparison",
        "",
        markdown_table(comparison_rows, ["model", "objective", "threshold", "roc_auc", "pr_auc", "precision", "recall", "f1"]),
    ]
    (BASE_DIR / "deep_learning_models_report_renewed.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(pd.DataFrame(comparison_rows).to_string(index=False))


if __name__ == "__main__":
    main()
