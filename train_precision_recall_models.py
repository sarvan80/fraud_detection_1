from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FOLDER = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
TRAIN_PATH = FOLDER / "train_login_model.csv"
TEST_PATH = FOLDER / "test_login_model.csv"

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


def evaluate(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    report = classification_report(
        y_true, prediction, output_dict=True, zero_division=0
    )
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "false_positive_count": int(((y_true == 0) & (prediction == 1)).sum()),
        "false_negative_count": int(((y_true == 1) & (prediction == 0)).sum()),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def choose_threshold(y_validation, probability, objective):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        result = evaluate(y_validation, probability, threshold)
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
        # Optimize alert purity; this model is intended for high-confidence action.
        selected = table.sort_values(
            ["precision", "f1", "recall"], ascending=False
        ).iloc[0]
    else:
        # Optimize fraud coverage; this model is intended for broad investigation.
        selected = table.sort_values(
            ["recall", "f1", "precision"], ascending=False
        ).iloc[0]
    return float(selected["threshold"]), table


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_columns = [
        c for c in train.columns if c not in LEAKAGE_COLUMNS | {"is_fraud"}
    ]

    x = add_domain_features(train[feature_columns])
    x_test = add_domain_features(test[feature_columns])
    y = train["is_fraud"].astype(int)
    y_test = test["is_fraud"].astype(int)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )

    precision_classifier = ExtraTreesClassifier(
        n_estimators=800,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    recall_classifier = RandomForestClassifier(
        n_estimators=800,
        max_features=None,
        min_samples_leaf=1,
        class_weight={0: 1, 1: 12},
        n_jobs=-1,
        random_state=43,
    )

    models = {
        "precision_optimized_model": (precision_classifier, "precision"),
        "recall_optimized_model": (recall_classifier, "recall"),
    }

    results = {}
    threshold_tables = {}
    test_probabilities = {}

    for name, (model, objective) in models.items():
        model.fit(x_train, y_train)
        validation_probability = model.predict_proba(x_validation)[:, 1]
        threshold, threshold_table = choose_threshold(
            y_validation, validation_probability, objective
        )
        test_probability = model.predict_proba(x_test)[:, 1]
        result = evaluate(y_test, test_probability, threshold)
        result["optimization_objective"] = objective
        result["validation_rows"] = int(len(x_validation))
        results[name] = result
        threshold_tables[name] = threshold_table
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

    comparison = pd.DataFrame([
        {
            "model": name,
            "objective": result["optimization_objective"],
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
    comparison.to_csv(FOLDER / "precision_recall_model_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    for name, probability in test_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probability)
        plt.plot(
            recall,
            precision,
            label=f"{name} AP={results[name]['pr_auc']:.3f}",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER / "precision_recall_models_curve.png", dpi=160)
    plt.close()

    summary = {
        "models": results,
        "feature_count": int(x.shape[1]),
        "engineered_features": [
            c for c in x.columns if c not in feature_columns
        ],
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
    }
    (FOLDER / "precision_recall_models_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Precision-Optimized and Recall-Optimized Models",
        "",
        "Both models use the same untouched test split. Thresholds are selected on a separate validation split.",
        "",
        "## Model comparison",
        "",
        comparison.to_string(index=False),
        "",
    ]
    for name, result in results.items():
        matrix = pd.DataFrame(
            result["confusion_matrix"],
            index=["actual_legitimate", "actual_fraud"],
            columns=["predicted_legitimate", "predicted_fraud"],
        )
        report.extend([
            f"## {name}",
            "",
            f"- Optimization objective: {result['optimization_objective']}",
            f"- Threshold: {result['threshold']:.3f}",
            f"- Precision: {result['precision']:.4f}",
            f"- Recall: {result['recall']:.4f}",
            f"- F1: {result['f1']:.4f}",
            f"- PR-AUC: {result['pr_auc']:.4f}",
            f"- False positives: {result['false_positive_count']}",
            f"- False negatives: {result['false_negative_count']}",
            "",
            matrix.to_string(),
            "",
        ])
    report.extend([
        "## Recommended usage",
        "",
        "Use `precision_optimized_model` when analyst capacity and false-alert cost are the primary constraints. Use `recall_optimized_model` when missed fraud is more costly and the organization can absorb additional alerts for investigation.",
    ])
    (FOLDER / "precision_recall_models_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(comparison.to_string(index=False))
    print("Created separately named precision and recall model artifacts.")


if __name__ == "__main__":
    main()
