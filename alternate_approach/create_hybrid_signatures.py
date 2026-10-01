from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
OUT = ROOT / "alternate_approach"
TRAIN_PATH = ROOT / "train_login_model.csv"
TEST_PATH = ROOT / "test_login_model.csv"


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


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = [c for c in train.columns if c not in LEAKAGE_COLUMNS | {"is_fraud"}]
    x_train = train[features].copy()
    x_test = test[features].copy()
    for frame in (x_train, x_test):
        for column in frame.columns:
            if frame[column].dtype == bool:
                frame[column] = frame[column].astype(int)
    return train, test, features, x_train.apply(pd.to_numeric, errors="coerce").fillna(0), x_test.apply(pd.to_numeric, errors="coerce").fillna(0)


def extract_tree_signatures(tree, features):
    tree_ = tree.tree_
    signatures = []

    def walk(node, conditions):
        left = tree_.children_left[node]
        right = tree_.children_right[node]
        if left == right:
            counts = tree_.value[node][0]
            total = counts.sum()
            probability = counts[1] / total if total else 0.0
            signatures.append({
                "signature_id": f"RULE-{len(signatures) + 1:03d}",
                "signature_type": "decision_tree_rule",
                "conditions": " AND ".join(conditions) if conditions else "all records",
                "predicted_fraud_probability": probability,
                "training_support": int(total),
                "tree_node": int(node),
            })
            return

        feature = features[tree_.feature[node]]
        threshold = tree_.threshold[node]
        walk(left, conditions + [f"{feature} <= {threshold:.4f}"])
        walk(right, conditions + [f"{feature} > {threshold:.4f}"])

    walk(0, [])
    return signatures


def apply_tree_leaf_rules(tree, x):
    leaves = tree.apply(x)
    leaf_to_rule = {}
    signatures = extract_tree_signatures(tree, list(x.columns))
    for signature in signatures:
        leaf_to_rule[signature["tree_node"]] = signature["signature_id"]
    return np.array([leaf_to_rule[leaf] for leaf in leaves])


def main():
    train, test, features, x_train, x_test = load_data()
    y_train = train["is_fraud"].astype(int).to_numpy()
    y_test = test["is_fraud"].astype(int).to_numpy()

    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(x_train, y_train)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    pca = PCA(n_components=min(8, x_train.shape[1]), random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
    train_clusters = kmeans.fit_predict(train_pca)
    test_clusters = kmeans.predict(test_pca)
    cluster_stats = pd.DataFrame({"cluster": train_clusters, "is_fraud": y_train}).groupby("cluster").agg(
        rows=("is_fraud", "size"),
        fraud_count=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean"),
    )
    global_rate = y_train.mean()
    cluster_stats["risk_lift"] = cluster_stats["fraud_rate"] / max(global_rate, 1e-9)

    isolation = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
    isolation.fit(train_scaled[y_train == 0])
    raw_train = -isolation.score_samples(train_scaled)
    raw_test = -isolation.score_samples(test_scaled)
    low, high = np.percentile(raw_train, [1, 99])
    anomaly_train = np.clip((raw_train - low) / max(high - low, 1e-9), 0, 1)
    anomaly_test = np.clip((raw_test - low) / max(high - low, 1e-9), 0, 1)

    rule_signatures = extract_tree_signatures(tree, features)
    train_rule_ids = apply_tree_leaf_rules(tree, x_train)
    test_rule_ids = apply_tree_leaf_rules(tree, x_test)
    tree_prob_train = tree.predict_proba(x_train)[:, 1]
    tree_prob_test = tree.predict_proba(x_test)[:, 1]

    cluster_risk_train = np.array([
        min(cluster_stats.loc[c, "risk_lift"] / 4, 1) for c in train_clusters
    ])
    cluster_risk_test = np.array([
        min(cluster_stats.loc[c, "risk_lift"] / 4, 1) for c in test_clusters
    ])
    hybrid_train = 0.65 * tree_prob_train + 0.20 * anomaly_train + 0.15 * cluster_risk_train
    hybrid_test = 0.65 * tree_prob_test + 0.20 * anomaly_test + 0.15 * cluster_risk_test

    # Match the operating threshold selected by the hybrid model pipeline.
    threshold = 0.75
    signature_rows = []
    for signature in rule_signatures:
        signature_id = signature["signature_id"]
        train_mask = train_rule_ids == signature_id
        test_mask = test_rule_ids == signature_id
        test_count = int(test_mask.sum())
        fraud_count = int(y_test[test_mask].sum())
        precision = fraud_count / test_count if test_count else 0.0
        recall = fraud_count / max(int(y_test.sum()), 1)
        signature_rows.append({
            **signature,
            "test_support": test_count,
            "test_fraud_count": fraud_count,
            "test_precision": precision,
            "test_recall": recall,
            "test_fraud_rate_pct": precision * 100,
            "operationally_selected": bool(signature["predicted_fraud_probability"] >= 0.5),
        })

    signatures = pd.DataFrame(signature_rows).sort_values(
        ["test_precision", "test_support"], ascending=False
    )
    signatures.to_csv(OUT / "hybrid_rule_signatures.csv", index=False)

    # Add anomaly and cluster signatures to the operational catalog.
    catalog_rows = signatures.to_dict(orient="records")
    anomaly_cutoff = float(np.quantile(anomaly_train, 0.99))
    anomaly_test_mask = anomaly_test >= anomaly_cutoff
    anomaly_fraud_count = int(y_test[anomaly_test_mask].sum())
    anomaly_support = int(anomaly_test_mask.sum())
    catalog_rows.append({
        "signature_id": "ANOMALY-001",
        "signature_type": "isolation_forest_anomaly",
        "conditions": f"anomaly_score >= {anomaly_cutoff:.4f}",
        "predicted_fraud_probability": None,
        "training_support": int((anomaly_train >= anomaly_cutoff).sum()),
        "tree_node": None,
        "test_support": anomaly_support,
        "test_fraud_count": anomaly_fraud_count,
        "test_precision": anomaly_fraud_count / anomaly_support if anomaly_support else 0.0,
        "test_recall": anomaly_fraud_count / max(int(y_test.sum()), 1),
        "test_fraud_rate_pct": 100 * anomaly_fraud_count / anomaly_support if anomaly_support else 0.0,
        "operationally_selected": True,
    })

    high_cluster = int(cluster_stats["risk_lift"].idxmax())
    cluster_test_mask = test_clusters == high_cluster
    cluster_fraud_count = int(y_test[cluster_test_mask].sum())
    cluster_support = int(cluster_test_mask.sum())
    catalog_rows.append({
        "signature_id": "CLUSTER-001",
        "signature_type": "high_risk_cluster",
        "conditions": f"cluster_id == {high_cluster}",
        "predicted_fraud_probability": float(cluster_stats.loc[high_cluster, "fraud_rate"]),
        "training_support": int(cluster_stats.loc[high_cluster, "rows"]),
        "tree_node": None,
        "test_support": cluster_support,
        "test_fraud_count": cluster_fraud_count,
        "test_precision": cluster_fraud_count / cluster_support if cluster_support else 0.0,
        "test_recall": cluster_fraud_count / max(int(y_test.sum()), 1),
        "test_fraud_rate_pct": 100 * cluster_fraud_count / cluster_support if cluster_support else 0.0,
        "operationally_selected": True,
    })

    catalog = pd.DataFrame(catalog_rows).sort_values(
        ["test_precision", "test_support"], ascending=False
    )
    catalog.to_csv(OUT / "hybrid_signature_catalog.csv", index=False)

    # Record which signatures fired on each held-out record.
    scored = test.copy()
    scored["tree_signature_id"] = test_rule_ids
    scored["cluster_id"] = test_clusters
    scored["anomaly_score"] = anomaly_test
    scored["hybrid_probability"] = hybrid_test
    scored["hybrid_predicted_fraud"] = (hybrid_test >= threshold).astype(int)
    scored["anomaly_signature_fired"] = anomaly_test_mask.astype(int)
    scored["high_risk_cluster_signature_fired"] = cluster_test_mask.astype(int)
    scored.to_csv(OUT / "hybrid_signature_matches_test.csv", index=False)

    report = {
        "signature_count": int(len(catalog)),
        "decision_tree_signature_count": int(len(signatures)),
        "anomaly_cutoff": anomaly_cutoff,
        "high_risk_cluster": high_cluster,
        "hybrid_threshold": threshold,
        "top_signatures": catalog.head(15).to_dict(orient="records"),
        "notes": [
            "Decision-tree signatures are extracted from interpretable conjunctions.",
            "Anomaly-001 flags events in the top 1% of the legitimate-training anomaly distribution.",
            "CLUSTER-001 flags the cluster with the highest training fraud lift.",
            "Signature precision and recall are measured on the held-out test set.",
        ],
    }
    (OUT / "hybrid_signature_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Hybrid Detection Signatures",
        "",
        "This catalog combines interpretable decision-tree rules, an Isolation Forest anomaly signature, and a high-risk cluster signature.",
        "",
        "## Signature types",
        "",
        "- Decision-tree signatures: conjunctions of feature thresholds extracted from the rule model.",
        "- ANOMALY-001: events above the 99th percentile of the legitimate-training anomaly score.",
        f"- CLUSTER-001: events assigned to cluster {high_cluster}, which has the highest training fraud lift.",
        "",
        "## Top signatures on held-out test data",
        "",
        catalog.head(15)[[
            "signature_id", "signature_type", "conditions",
            "test_support", "test_fraud_count", "test_precision", "test_recall"
        ]].to_string(index=False),
        "",
        "## Governance guidance",
        "",
        "Promote a signature only after reviewing its support, precision, recall, stability over time, and analyst impact. Keep the extracted condition text, model version, training window, owner, approval state, and retirement criteria with each production signature.",
    ]
    (OUT / "hybrid_signatures_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Created {len(catalog)} signatures.")
    print(f"Top signature: {catalog.iloc[0]['signature_id']} ({catalog.iloc[0]['test_precision']:.3f} precision)")
    print("Created hybrid signature catalog and held-out match results.")


if __name__ == "__main__":
    main()
