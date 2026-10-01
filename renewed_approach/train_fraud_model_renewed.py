from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from renewed_modeling_utils import (
    BASE_DIR,
    EXTENDED_LEAKAGE_COLUMNS,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    evaluate_predictions,
    markdown_table,
    save_confusion_matrix_artifacts,
    save_feature_importance,
    save_pr_curve,
    threshold_search,
    write_json,
)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_columns = [column for column in train.columns if column not in EXTENDED_LEAKAGE_COLUMNS | {TARGET}]

    X_train = train[feature_columns].copy()
    y_train = train[TARGET].astype(int)
    X_test = test[feature_columns].copy()
    y_test = test[TARGET].astype(int)

    models = {
        "logistic_regression_renewed": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "random_forest_renewed": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=450,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
    }

    results = {}
    fitted_models = {}
    probabilities_by_model = {}

    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        probabilities_by_model[name] = probabilities
        results[name] = evaluate_predictions(y_test, probabilities, 0.5)
        fitted_models[name] = estimator

        threshold_table = threshold_search(y_test, probabilities, np.linspace(0.05, 0.95, 181))
        threshold_table.to_csv(BASE_DIR / f"{name}_threshold_search.csv", index=False)

    best_name = max(results, key=lambda name: results[name]["pr_auc"])
    best_model = fitted_models[best_name]
    best_probabilities = probabilities_by_model[best_name]
    best_predictions = (best_probabilities >= 0.5).astype(int)

    scored = test.copy()
    scored["fraud_probability_renewed"] = best_probabilities
    scored["predicted_fraud_renewed"] = best_predictions
    scored.to_csv(BASE_DIR / "scored_test_login_renewed.csv", index=False)

    save_confusion_matrix_artifacts(
        results[best_name]["confusion_matrix"],
        BASE_DIR / "confusion_matrix_renewed.csv",
        BASE_DIR / "confusion_matrix_renewed.png",
        f"Renewed baseline confusion matrix - {best_name}",
    )

    model_step = best_model.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importance_values = model_step.feature_importances_
    else:
        importance_values = np.abs(model_step.coef_[0])
    importance = save_feature_importance(
        importance_values,
        feature_columns,
        BASE_DIR / "feature_importance_renewed.csv",
        BASE_DIR / "feature_importance_renewed.png",
        f"Renewed baseline feature importance - {best_name}",
    )

    save_pr_curve(
        [(best_name, y_test.to_numpy(), best_probabilities)],
        BASE_DIR / "precision_recall_curve_renewed.png",
        "Renewed baseline precision-recall curve",
    )

    summary = {
        "selected_model": best_name,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "excluded_leakage_columns": sorted(EXTENDED_LEAKAGE_COLUMNS),
        "feature_count": len(feature_columns),
        "model_results": results,
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    write_json(BASE_DIR / "model_metrics_renewed.json", summary)

    comparison_rows = [
        {
            "model": name,
            "roc_auc": metric["roc_auc"],
            "pr_auc": metric["pr_auc"],
            "precision": metric["precision"],
            "recall": metric["recall"],
            "f1": metric["f1"],
            "threshold": metric["threshold"],
        }
        for name, metric in results.items()
    ]
    report_lines = [
        "# Renewed Baseline Fraud Model",
        "",
        "This script mirrors the original baseline approach on the enriched renewed dataset, comparing logistic regression and random forest with the default 0.50 operating threshold.",
        "",
        "## Model comparison",
        "",
        markdown_table(comparison_rows, ["model", "roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]),
        "",
        "## Selected model",
        "",
        f"- Model: **{best_name}**",
        f"- ROC-AUC: **{results[best_name]['roc_auc']:.4f}**",
        f"- PR-AUC: **{results[best_name]['pr_auc']:.4f}**",
        f"- Precision: **{results[best_name]['precision']:.4f}**",
        f"- Recall: **{results[best_name]['recall']:.4f}**",
        f"- F1: **{results[best_name]['f1']:.4f}**",
        "",
        "## Highest-ranked features",
        "",
        markdown_table(importance.head(15).to_dict(orient="records"), ["feature", "importance"]),
    ]
    (BASE_DIR / "model_results_renewed.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Selected model: {best_name}")
    print(f"PR-AUC: {results[best_name]['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
