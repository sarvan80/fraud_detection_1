from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FOLDER = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
TRAIN_PATH = FOLDER / "train_login_model.csv"
TEST_PATH = FOLDER / "test_login_model.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

target = "is_fraud"

# Exclude labels or post-outcome aggregates to avoid target leakage.
leakage_columns = {
    "txn_fraud_count",
    "user_prev_fraud_count",
    "txn_count",
    "avg_txn_amount",
    "max_txn_amount",
    "high_risk_txn_rate",
    "merchant_risk_txn_count",
    "beneficiary_mismatch_count",
}
feature_columns = [c for c in train.columns if c != target and c not in leakage_columns]

X_train = train[feature_columns].copy()
y_train = train[target].astype(int)
X_test = test[feature_columns].copy()
y_test = test[target].astype(int)

# Treat boolean indicator columns as numeric model inputs.
for frame in (X_train, X_test):
    bool_columns = frame.select_dtypes(include=["bool"]).columns
    frame[bool_columns] = frame[bool_columns].astype(int)

categorical_columns = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_columns = [c for c in feature_columns if c not in categorical_columns]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_columns,
        ),
    ]
)

models = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    ),
}

results = {}
fitted_models = {}

for name, estimator in models.items():
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    results[name] = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    fitted_models[name] = pipeline

best_name = max(results, key=lambda name: results[name]["pr_auc"])
best_model = fitted_models[best_name]
best_probabilities = best_model.predict_proba(X_test)[:, 1]
best_predictions = (best_probabilities >= 0.5).astype(int)

# Export scored test rows for review.
scored = test.copy()
scored["fraud_probability"] = best_probabilities
scored["predicted_fraud"] = best_predictions
scored.to_csv(FOLDER / "scored_test_login.csv", index=False)

# Export feature importance / coefficients.
fitted_preprocessor = best_model.named_steps["preprocessor"]
model = best_model.named_steps["model"]
transformed_names = fitted_preprocessor.get_feature_names_out()

if hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_
else:
    importance_values = abs(model.coef_[0])

importance = pd.DataFrame(
    {"feature": transformed_names, "importance": importance_values}
).sort_values("importance", ascending=False)
importance.to_csv(FOLDER / "feature_importance.csv", index=False)

# Export a compact confusion matrix CSV.
matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1])
pd.DataFrame(
    matrix,
    index=["actual_legitimate", "actual_fraud"],
    columns=["predicted_legitimate", "predicted_fraud"],
).to_csv(FOLDER / "confusion_matrix.csv")

# Create a precision-recall curve for the selected model.
precision, recall, thresholds = precision_recall_curve(y_test, best_probabilities)
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, label=f"{best_name} (AP={results[best_name]['pr_auc']:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Fraud Detection Precision-Recall Curve")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(FOLDER / "precision_recall_curve.png", dpi=160)
plt.close()

# Create a feature importance chart.
top = importance.head(15).sort_values("importance")
plt.figure(figsize=(9, 6))
plt.barh(top["feature"], top["importance"])
plt.xlabel("Importance")
plt.title(f"Top Fraud Features - {best_name}")
plt.tight_layout()
plt.savefig(FOLDER / "feature_importance.png", dpi=160)
plt.close()

summary = {
    "selected_model": best_name,
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "train_fraud_rate": float(y_train.mean()),
    "test_fraud_rate": float(y_test.mean()),
    "excluded_leakage_columns": sorted(leakage_columns),
    "model_results": results,
}
(FOLDER / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"Selected model: {best_name}")
for name, metric in results.items():
    print(
        f"{name}: ROC-AUC={metric['roc_auc']:.4f}, "
        f"PR-AUC={metric['pr_auc']:.4f}, "
        f"precision={metric['precision']:.4f}, "
        f"recall={metric['recall']:.4f}, "
        f"F1={metric['f1']:.4f}"
    )
print("Created scored_test_login.csv, feature_importance.csv, confusion_matrix.csv, model_metrics.json")
