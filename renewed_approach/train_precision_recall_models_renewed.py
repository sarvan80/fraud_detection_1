from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

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


def choose_threshold(y_validation: pd.Series, probability: np.ndarray, objective: str) -> tuple[float, pd.DataFrame]:
    table = threshold_search(y_validation, probability, np.linspace(0.05, 0.95, 181))
    if objective == "precision":
        selected = table.sort_values(["precision", "f1", "recall"], ascending=False).iloc[0]
    else:
        selected = table.sort_values(["recall", "f1", "precision"], ascending=False).iloc[0]
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

    models = {
        "precision_optimized_model_renewed": (
            ExtraTreesClassifier(
                n_estimators=900,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            "precision",
        ),
        "recall_optimized_model_renewed": (
            RandomForestClassifier(
                n_estimators=900,
                max_features=None,
                min_samples_leaf=1,
                class_weight={0: 1, 1: 14},
                n_jobs=-1,
                random_state=43,
            ),
            "recall",
        ),
    }

    results = {}
    threshold_tables = {}
    test_probabilities = {}
    feature_importance_rows = []

    for name, (model, objective) in models.items():
        model.fit(X_train, y_train)
        validation_probability = model.predict_proba(X_validation)[:, 1]
        threshold, threshold_table = choose_threshold(y_validation, validation_probability, objective)
        test_probability = model.predict_proba(X_test)[:, 1]
        result = evaluate_predictions(y_test, test_probability, threshold)
        result["optimization_objective"] = objective
        result["validation_rows"] = int(len(X_validation))
        results[name] = result
        threshold_tables[name] = threshold_table
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
        importance = save_feature_importance(
            model.feature_importances_,
            list(X_full.columns),
            BASE_DIR / f"{name}_feature_importance.csv",
            BASE_DIR / f"{name}_feature_importance.png",
            f"{name} feature importance",
        )
        feature_importance_rows.append((name, importance))

    comparison_rows = [
        {
            "model": name,
            "objective": result["optimization_objective"],
            "threshold": result["threshold"],
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        }
        for name, result in results.items()
    ]
    pd.DataFrame(comparison_rows).to_csv(BASE_DIR / "precision_recall_model_comparison_renewed.csv", index=False)

    save_pr_curve(
        [(name, y_test.to_numpy(), probability) for name, probability in test_probabilities.items()],
        BASE_DIR / "precision_recall_models_curve_renewed.png",
        "Renewed precision/recall models comparison",
    )

    summary = {
        "models": results,
        "feature_count": int(X_full.shape[1]),
        "engineered_features": [column for column in X_full.columns if column not in feature_columns],
        "excluded_leakage_columns": sorted(EXTENDED_LEAKAGE_COLUMNS),
    }
    write_json(BASE_DIR / "precision_recall_models_metrics_renewed.json", summary)

    report_lines = [
        "# Renewed Precision-Optimized and Recall-Optimized Models",
        "",
        "Both renewed models use the enriched feature set, choose thresholds only on validation data, and are evaluated once on the untouched test split.",
        "",
        "## Model comparison",
        "",
        markdown_table(comparison_rows, ["model", "objective", "threshold", "roc_auc", "pr_auc", "precision", "recall", "f1"]),
        "",
    ]
    for name, importance in feature_importance_rows:
        report_lines.extend([
            f"## {name}",
            "",
            markdown_table(
                [{
                    "threshold": results[name]["threshold"],
                    "roc_auc": results[name]["roc_auc"],
                    "pr_auc": results[name]["pr_auc"],
                    "precision": results[name]["precision"],
                    "recall": results[name]["recall"],
                    "f1": results[name]["f1"],
                }],
                ["threshold", "roc_auc", "pr_auc", "precision", "recall", "f1"],
            ),
            "",
            markdown_table(importance.head(15).to_dict(orient="records"), ["feature", "importance"]),
            "",
        ])
    (BASE_DIR / "precision_recall_models_report_renewed.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(pd.DataFrame(comparison_rows).to_string(index=False))


if __name__ == "__main__":
    main()
