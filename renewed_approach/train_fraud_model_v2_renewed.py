from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance

from renewed_modeling_utils import (
    BASE_DIR,
    EXTENDED_LEAKAGE_COLUMNS,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    add_domain_features,
    evaluate_predictions,
    markdown_table,
    save_confusion_matrix_artifacts,
    save_feature_importance,
    save_pr_curve,
    threshold_search,
    write_json,
)
from sklearn.model_selection import train_test_split


def select_threshold(y_validation: pd.Series, validation_probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    table = threshold_search(y_validation, validation_probability, np.linspace(0.05, 0.95, 181))
    eligible = table[table["recall"] >= 0.70]
    selected = (eligible if not eligible.empty else table).sort_values(["f1", "precision"], ascending=False).iloc[0]
    return float(selected["threshold"]), table


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_columns = [column for column in train.columns if column not in EXTENDED_LEAKAGE_COLUMNS | {TARGET}]

    X_full = add_domain_features(train[feature_columns])
    X_test = add_domain_features(test[feature_columns])
    y_full = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_full,
        y_full,
        test_size=0.25,
        stratify=y_full,
        random_state=42,
    )

    candidates = {
        "random_forest_v2_renewed": RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "extra_trees_v2_renewed": ExtraTreesClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "hist_gradient_boosting_v2_renewed": HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.05,
            max_leaf_nodes=21,
            l2_regularization=1.2,
            random_state=42,
        ),
    }

    results = {}
    fitted = {}
    threshold_tables = {}
    test_probabilities = {}

    for name, model in candidates.items():
        if "hist_gradient" in name:
            sample_weight = np.where(y_train.to_numpy() == 1, 8.5, 1.0)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        validation_probability = model.predict_proba(X_validation)[:, 1]
        threshold, threshold_table = select_threshold(y_validation, validation_probability)
        test_probability = model.predict_proba(X_test)[:, 1]

        results[name] = evaluate_predictions(y_test, test_probability, threshold)
        results[name]["validation_threshold_selection"] = True
        results[name]["validation_rows"] = int(len(X_validation))
        fitted[name] = model
        threshold_tables[name] = threshold_table
        test_probabilities[name] = test_probability
        threshold_table.to_csv(BASE_DIR / f"{name}_threshold_search.csv", index=False)

    best_name = max(results, key=lambda name: (results[name]["pr_auc"], results[name]["f1"]))
    best_model = fitted[best_name]
    best_threshold = results[best_name]["threshold"]
    best_probability = test_probabilities[best_name]

    scored = test.copy()
    scored["fraud_probability_v2_renewed"] = best_probability
    scored["predicted_fraud_v2_renewed"] = (best_probability >= best_threshold).astype(int)
    scored.to_csv(BASE_DIR / "scored_test_login_v2_renewed.csv", index=False)

    save_confusion_matrix_artifacts(
        results[best_name]["confusion_matrix"],
        BASE_DIR / "confusion_matrix_v2_renewed.csv",
        BASE_DIR / "confusion_matrix_v2_renewed.png",
        f"Renewed v2 confusion matrix - {best_name}",
    )

    importance_values = getattr(best_model, "feature_importances_", None)
    if importance_values is None:
        importance_values = permutation_importance(
            best_model,
            X_validation,
            y_validation,
            n_repeats=5,
            random_state=42,
            scoring="average_precision",
        ).importances_mean
    importance = save_feature_importance(
        importance_values,
        list(X_full.columns),
        BASE_DIR / "feature_importance_v2_renewed.csv",
        BASE_DIR / "feature_importance_v2_renewed.png",
        f"Renewed v2 feature importance - {best_name}",
    )

    save_pr_curve(
        [(name, y_test.to_numpy(), probability) for name, probability in test_probabilities.items()],
        BASE_DIR / "precision_recall_curve_v2_renewed.png",
        "Renewed v2 precision-recall comparison",
    )

    comparison_rows = [
        {
            "model": name,
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "threshold": result["threshold"],
        }
        for name, result in results.items()
    ]
    pd.DataFrame(comparison_rows).to_csv(BASE_DIR / "model_comparison_v2_renewed.csv", index=False)

    summary = {
        "version": "v2_renewed",
        "selected_model": best_name,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "excluded_leakage_columns": sorted(EXTENDED_LEAKAGE_COLUMNS),
        "engineered_features": [column for column in X_full.columns if column not in feature_columns],
        "candidate_results": results,
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    write_json(BASE_DIR / "model_metrics_v2_renewed.json", summary)

    report_lines = [
        "# Renewed Fraud Model Version 2 Results",
        "",
        "The renewed v2 pipeline adds cross-surface device, identity, network, and transaction interaction features while preserving validation-only threshold selection and untouched test reporting.",
        "",
        "## Model comparison",
        "",
        markdown_table(comparison_rows, ["model", "roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]),
        "",
        "## Selected model",
        "",
        f"- Model: **{best_name}**",
        f"- Threshold: **{best_threshold:.4f}**",
        f"- ROC-AUC: **{results[best_name]['roc_auc']:.4f}**",
        f"- PR-AUC: **{results[best_name]['pr_auc']:.4f}**",
        f"- Precision: **{results[best_name]['precision']:.4f}**",
        f"- Recall: **{results[best_name]['recall']:.4f}**",
        f"- F1: **{results[best_name]['f1']:.4f}**",
        "",
        "## Highest-ranked features",
        "",
        markdown_table(importance.head(20).to_dict(orient="records"), ["feature", "importance"]),
    ]
    (BASE_DIR / "model_results_v2_renewed.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Selected model: {best_name}")
    print(f"PR-AUC: {results[best_name]['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
