from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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
    result["behavior_irregularity"] = (
        result["mouse_velocity"] / (result["typing_velocity"] + 1.0)
    )
    result["interaction_density"] = (
        result["click_count"] + result["keystroke_count"]
    ) / (result["user_login_count"] + 1.0)
    result["network_travel_interaction"] = (
        result["distance_from_home_km"] * (1 + result["is_proxy"] + result["is_vpn"])
    )
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


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
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": report,
    }


def select_threshold(y_validation, validation_probability):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        metric = evaluate(y_validation, validation_probability, threshold)
        rows.append({
            "threshold": threshold,
            "precision": metric["precision"],
            "recall": metric["recall"],
            "f1": metric["f1"],
        })
    table = pd.DataFrame(rows)
    eligible = table[table["recall"] >= 0.70]
    selected = (eligible if not eligible.empty else table).sort_values(
        ["f1", "precision"], ascending=False
    ).iloc[0]
    return float(selected["threshold"]), table


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    target = "is_fraud"

    feature_columns = [
        c for c in train.columns
        if c not in LEAKAGE_COLUMNS | {target}
    ]
    x = add_domain_features(train[feature_columns])
    x_test = add_domain_features(test[feature_columns])
    y = train[target].astype(int)
    y_test = test[target].astype(int)

    # Keep the final test set untouched while selecting thresholds on validation data.
    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )

    candidates = {
        "random_forest_v2": RandomForestClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "extra_trees_v2": ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "hist_gradient_boosting_v2": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    results = {}
    fitted = {}
    threshold_tables = {}

    for name, model in candidates.items():
        if name == "hist_gradient_boosting_v2":
            sample_weight = np.where(y_train.to_numpy() == 1, 8.0, 1.0)
            model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(x_train, y_train)

        validation_probability = model.predict_proba(x_validation)[:, 1]
        threshold, threshold_table = select_threshold(
            y_validation, validation_probability
        )
        test_probability = model.predict_proba(x_test)[:, 1]
        result = evaluate(y_test, test_probability, threshold)
        result["validation_threshold_selection"] = True
        result["validation_rows"] = int(len(x_validation))
        results[name] = result
        fitted[name] = model
        threshold_tables[name] = threshold_table

    best_name = max(
        results,
        key=lambda name: (results[name]["pr_auc"], results[name]["f1"]),
    )
    best_model = fitted[best_name]
    best_threshold = results[best_name]["threshold"]
    best_probability = best_model.predict_proba(x_test)[:, 1]

    scored = test.copy()
    scored["fraud_probability_v2"] = best_probability
    scored["predicted_fraud_v2"] = (
        best_probability >= best_threshold
    ).astype(int)
    scored.to_csv(FOLDER / "scored_test_login_v2.csv", index=False)

    pd.DataFrame(
        results[best_name]["confusion_matrix"],
        index=["actual_legitimate", "actual_fraud"],
        columns=["predicted_legitimate", "predicted_fraud"],
    ).to_csv(FOLDER / "confusion_matrix_v2.csv")

    importance = pd.DataFrame({
        "feature": x.columns,
        "importance": getattr(
            best_model,
            "feature_importances_",
            np.zeros(len(x.columns)),
        ),
    }).sort_values("importance", ascending=False)
    importance.to_csv(FOLDER / "feature_importance_v2.csv", index=False)

    top = importance.head(15).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"], color="#244a73")
    plt.title(f"Top Features - {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FOLDER / "feature_importance_v2.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, best_probability)
    plt.figure(figsize=(8, 5))
    plt.plot(
        recall,
        precision,
        color="#d9534f",
        label=f"{best_name} (AP={results[best_name]['pr_auc']:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Version 2 precision-recall curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER / "precision_recall_curve_v2.png", dpi=160)
    plt.close()

    baseline = json.loads(
        (FOLDER / "model_metrics.json").read_text(encoding="utf-8")
    )
    baseline_name = baseline["selected_model"]
    baseline_metric = baseline["model_results"][baseline_name]
    comparison_rows = [{
        "model": f"Prior baseline ({baseline_name})",
        **{key: baseline_metric[key] for key in [
            "roc_auc", "pr_auc", "precision", "recall", "f1"
        ]},
        "threshold": 0.5,
    }]
    for name, result in results.items():
        comparison_rows.append({
            "model": name,
            **{key: result[key] for key in [
                "roc_auc", "pr_auc", "precision", "recall", "f1"
            ]},
            "threshold": result["threshold"],
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(FOLDER / "model_comparison_v2.csv", index=False)

    for name, table in threshold_tables.items():
        table.to_csv(FOLDER / f"{name}_threshold_search.csv", index=False)

    summary = {
        "version": "v2",
        "selected_model": best_name,
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "test_rows": int(len(x_test)),
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
        "engineered_features": [
            c for c in x.columns if c not in feature_columns
        ],
        "candidate_results": results,
        "comparison": comparison.to_dict(orient="records"),
        "metric_delta_best_v2_minus_baseline": {
            key: results[best_name][key] - baseline_metric[key]
            for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]
        },
    }
    (FOLDER / "model_metrics_v2.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Fraud Model Version 2 Results",
        "",
        "Version 2 adds domain-derived risk interactions, evaluates Extra Trees, Random Forest, and histogram gradient boosting, and selects the operating threshold using a stratified validation split rather than the test set.",
        "",
        "## Model comparison",
        "",
        comparison.to_string(index=False),
        "",
        "## Selected model",
        "",
        f"- Model: **{best_name}**",
        f"- Threshold: **{best_threshold:.3f}**",
        f"- ROC-AUC: **{results[best_name]['roc_auc']:.4f}**",
        f"- PR-AUC: **{results[best_name]['pr_auc']:.4f}**",
        f"- Precision: **{results[best_name]['precision']:.4f}**",
        f"- Recall: **{results[best_name]['recall']:.4f}**",
        f"- F1: **{results[best_name]['f1']:.4f}**",
        "",
        "## Confusion matrix",
        "",
        pd.DataFrame(
            results[best_name]["confusion_matrix"],
            index=["actual_legitimate", "actual_fraud"],
            columns=["predicted_legitimate", "predicted_fraud"],
        ).to_string(),
        "",
        "## Interpretation",
        "",
        "Metrics are measured on the same held-out test split as the original baseline. The validation split is used only for threshold selection. PR-AUC, fraud recall, precision, and F1 are emphasized because the fraud class is rare and accuracy alone can be misleading.",
    ]
    (FOLDER / "model_results_v2.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(f"Selected model: {best_name}")
    print(f"Results written to {FOLDER}")


if __name__ == "__main__":
    main()
