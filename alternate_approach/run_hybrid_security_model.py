from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
OUT = ROOT / "alternate_approach"
TRAIN_PATH = ROOT / "train_login_model.csv"
TEST_PATH = ROOT / "test_login_model.csv"


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    report = classification_report(y_true, prediction, output_dict=True, zero_division=0)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "classification_report": report,
    }


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    target = "is_fraud"

    # These fields are post-outcome or derived from the target in the generated data.
    leakage = {
        "txn_fraud_count",
        "user_prev_fraud_count",
        "txn_count",
        "avg_txn_amount",
        "max_txn_amount",
        "high_risk_txn_rate",
        "merchant_risk_txn_count",
        "beneficiary_mismatch_count",
    }
    features = [c for c in train.columns if c not in leakage | {target}]
    X_train = train[features].copy()
    X_test = test[features].copy()
    y_train = train[target].astype(int).to_numpy()
    y_test = test[target].astype(int).to_numpy()

    # Boolean columns in the model-ready files are converted to numeric indicators.
    for frame in (X_train, X_test):
        for column in frame.columns:
            if frame[column].dtype == bool:
                frame[column] = frame[column].astype(int)
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)

    # Unsupervised profile: compress the feature space and cluster training behavior.
    pca = PCA(n_components=min(8, X_train.shape[1]), random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
    train_clusters = kmeans.fit_predict(train_pca)
    test_clusters = kmeans.predict(test_pca)

    # Cluster-derived pseudo-label signal: clusters with elevated fraud prevalence are suspicious.
    cluster_stats = pd.DataFrame({"cluster": train_clusters, "is_fraud": y_train}).groupby("cluster").agg(
        rows=("is_fraud", "size"),
        fraud_count=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean"),
    )
    global_rate = y_train.mean()
    cluster_risk = ((cluster_stats["fraud_rate"] / max(global_rate, 1e-9)).clip(0, 4) / 4).to_dict()
    train_cluster_risk = np.array([cluster_risk[c] for c in train_clusters])
    test_cluster_risk = np.array([cluster_risk.get(c, 0.0) for c in test_clusters])

    # Anomaly component: learn the normal traffic profile from legitimate training events only.
    normal_mask = y_train == 0
    isolation = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    isolation.fit(train_scaled[normal_mask])
    raw_anomaly_train = -isolation.score_samples(train_scaled)
    raw_anomaly_test = -isolation.score_samples(test_scaled)
    low, high = np.percentile(raw_anomaly_train, [1, 99])
    anomaly_train = np.clip((raw_anomaly_train - low) / max(high - low, 1e-9), 0, 1)
    anomaly_test = np.clip((raw_anomaly_test - low) / max(high - low, 1e-9), 0, 1)

    # Interpretable rule extraction from discovered/confirmed classes.
    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_train, y_train)
    tree_train = tree.predict_proba(X_train)[:, 1]
    tree_test = tree.predict_proba(X_test)[:, 1]

    # Hybrid decision-intelligence score: rules + anomaly profile + cluster pseudo-class.
    train_probability = 0.65 * tree_train + 0.20 * anomaly_train + 0.15 * train_cluster_risk
    test_probability = 0.65 * tree_test + 0.20 * anomaly_test + 0.15 * test_cluster_risk

    # Select an operating threshold on training data, prioritizing F1 while retaining recall.
    candidates = np.linspace(0.10, 0.90, 161)
    candidate_rows = []
    for threshold in candidates:
        result = metrics(y_train, train_probability, threshold)
        candidate_rows.append({
            "threshold": threshold,
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        })
    threshold_table = pd.DataFrame(candidate_rows)
    eligible = threshold_table[threshold_table["recall"] >= 0.70]
    selected = (eligible if not eligible.empty else threshold_table).sort_values(
        ["f1", "precision"], ascending=False
    ).iloc[0]
    threshold = float(selected["threshold"])

    hybrid_metrics = metrics(y_test, test_probability, threshold)
    cluster_stats.to_csv(OUT / "hybrid_cluster_profile.csv")
    threshold_table.to_csv(OUT / "hybrid_threshold_search.csv", index=False)

    scored = test.copy()
    scored["cluster_id"] = test_clusters
    scored["cluster_risk"] = test_cluster_risk
    scored["anomaly_score"] = anomaly_test
    scored["rule_probability"] = tree_test
    scored["hybrid_probability"] = test_probability
    scored["predicted_fraud"] = (test_probability >= threshold).astype(int)
    scored.to_csv(OUT / "hybrid_scored_test.csv", index=False)

    cm = np.array(hybrid_metrics["confusion_matrix"])
    pd.DataFrame(
        cm,
        index=["actual_legitimate", "actual_fraud"],
        columns=["predicted_legitimate", "predicted_fraud"],
    ).to_csv(OUT / "hybrid_confusion_matrix.csv")

    rule_text = export_text(tree, feature_names=features)
    (OUT / "extracted_rules.txt").write_text(rule_text, encoding="utf-8")

    precision, recall, _ = precision_recall_curve(y_test, test_probability)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color="#d9534f", label=f"Hybrid PR-AUC={hybrid_metrics['pr_auc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Hybrid rule + anomaly + cluster precision-recall curve")
    plt.grid(alpha=0.25)
    plt.legend()
    save_plot(OUT / "hybrid_precision_recall_curve.png")

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["predicted legit", "predicted fraud"], rotation=20)
    plt.yticks([0, 1], ["actual legit", "actual fraud"])
    for row in range(2):
        for column in range(2):
            plt.text(column, row, cm[row, column], ha="center", va="center")
    plt.title("Hybrid confusion matrix")
    save_plot(OUT / "hybrid_confusion_matrix.png")

    baseline = json.loads((ROOT / "model_metrics.json").read_text(encoding="utf-8"))
    baseline_name = baseline["selected_model"]
    baseline_metrics = baseline["model_results"][baseline_name]
    comparison = pd.DataFrame([
        {
            "model": f"Prior baseline ({baseline_name})",
            "roc_auc": baseline_metrics["roc_auc"],
            "pr_auc": baseline_metrics["pr_auc"],
            "precision": baseline_metrics["precision"],
            "recall": baseline_metrics["recall"],
            "f1": baseline_metrics["f1"],
            "threshold": 0.5,
        },
        {
            "model": "Alternate hybrid",
            "roc_auc": hybrid_metrics["roc_auc"],
            "pr_auc": hybrid_metrics["pr_auc"],
            "precision": hybrid_metrics["precision"],
            "recall": hybrid_metrics["recall"],
            "f1": hybrid_metrics["f1"],
            "threshold": threshold,
        },
    ])
    comparison.to_csv(OUT / "model_comparison.csv", index=False)

    report = {
        "approach": "hybrid rule extraction + anomaly detection + clustering",
        "train_rows": len(train),
        "test_rows": len(test),
        "excluded_leakage_columns": sorted(leakage),
        "pca_components": pca.n_components_,
        "cluster_profile": cluster_stats.reset_index().to_dict(orient="records"),
        "selected_threshold": threshold,
        "prior_baseline": baseline_metrics,
        "hybrid_metrics": hybrid_metrics,
        "metric_delta_hybrid_minus_baseline": {
            key: hybrid_metrics[key] - baseline_metrics[key]
            for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]
        },
    }
    (OUT / "hybrid_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Alternate Hybrid Model Results",
        "",
        "This run applies the alternate PNG-based approach to the same train/test split used by the prior baseline.",
        "",
        "## Processing",
        "",
        "1. Excluded target-derived transaction and prior-fraud fields.",
        "2. Standardized numeric features.",
        "3. Reduced the feature space with PCA.",
        "4. Clustered behavior into four groups and converted cluster fraud prevalence into a pseudo-class risk signal.",
        "5. Fit Isolation Forest on legitimate training events to detect deviations from normal traffic.",
        "6. Extracted interpretable decision-tree rules.",
        "7. Combined rule probability, anomaly score, and cluster risk into a hybrid score.",
        "8. Selected a training threshold with recall constrained to at least 70%.",
        "",
        "## Test comparison",
        "",
        comparison.to_string(index=False),
        "",
        "## Hybrid confusion matrix",
        "",
        pd.DataFrame(
            cm,
            index=["actual_legitimate", "actual_fraud"],
            columns=["predicted_legitimate", "predicted_fraud"],
        ).to_string(),
        "",
        "## Interpretation",
        "",
        "The hybrid score is designed for explainability and discovery: the decision tree supplies explicit rules, Isolation Forest identifies departures from normal behavior, and clustering provides a pseudo-class signal when labels are sparse. Compare precision, recall, and PR-AUC together because the fraud class is rare.",
    ]
    (OUT / "hybrid_results_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(comparison.to_string(index=False))
    print(f"Hybrid threshold: {threshold:.3f}")
    print("Created hybrid model artifacts in", OUT)


if __name__ == "__main__":
    main()
