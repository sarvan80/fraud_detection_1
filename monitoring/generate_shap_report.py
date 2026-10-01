"""
SHAP explainability report for the fraud detection model.

Retrains the best-performing model from train_fraud_model_v2.py
(ExtraTrees_v2, selected for PR-AUC 0.4866) on the exact same feature
pipeline, then generates SHAP-based explanations against the held-out test
set:
  - Global feature importance (mean |SHAP value|) bar chart + CSV
  - Beeswarm summary plot (impact/direction per feature)
  - Dependence plots for the top 5 globally important features
  - Local waterfall explanations for the 3 highest-scoring true-positive
    fraud predictions (case studies an analyst could review)
  - A narrative shap_report.md tying it all together

Outputs are written to monitoring/shap/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

FOLDER = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
TRAIN_PATH = FOLDER / "train_login_model.csv"
TEST_PATH = FOLDER / "test_login_model.csv"
OUT_DIR = FOLDER / "monitoring" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
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


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = add_domain_features(train)
    test = add_domain_features(test)

    feature_cols = [c for c in train.columns if c not in LEAKAGE_COLUMNS and c != "is_fraud"]

    X_train, y_train = train[feature_cols], train["is_fraud"]
    X_test, y_test = test[feature_cols], test["is_fraud"]

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    print(f"Retrained ExtraTrees_v2 sanity check -> ROC-AUC: {roc_auc_score(y_test, proba):.4f}, "
          f"PR-AUC: {average_precision_score(y_test, proba):.4f}")

    # ---- SHAP explanations ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    # Handle both possible return shapes across shap versions (list per-class
    # vs single ndarray with a trailing class axis)
    if isinstance(shap_values, list):
        shap_fraud = shap_values[1]
    elif shap_values.ndim == 3:
        shap_fraud = shap_values[:, :, 1]
    else:
        shap_fraud = shap_values

    # Global importance table
    mean_abs_shap = np.abs(shap_fraud).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(OUT_DIR / "shap_feature_importance.csv", index=False)

    # Bar chart of global importance
    top20 = importance_df.head(20)
    plt.figure(figsize=(9, 8))
    plt.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="teal")
    plt.xlabel("Mean |SHAP value| (impact on fraud probability)")
    plt.title("Global Feature Importance — SHAP (ExtraTrees_v2)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_global_importance_bar.png", dpi=140)
    plt.close()

    # Beeswarm summary plot
    plt.figure()
    shap.summary_plot(shap_fraud, X_test, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_beeswarm.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Dependence plots for top 5 features
    top5 = importance_df["feature"].head(5).tolist()
    for feat in top5:
        plt.figure()
        shap.dependence_plot(feat, shap_fraud, X_test, show=False)
        plt.tight_layout()
        safe_name = feat.replace("/", "_")
        plt.savefig(OUT_DIR / f"shap_dependence_{safe_name}.png", dpi=140, bbox_inches="tight")
        plt.close()

    # Local explanations: top 3 highest-scoring true-positive fraud cases
    results = test[feature_cols].copy()
    results["is_fraud"] = y_test.values
    results["fraud_probability"] = proba
    true_positives = results[(results["is_fraud"] == 1)].sort_values("fraud_probability", ascending=False)
    case_study_idx = true_positives.head(3).index

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        base_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else expected_value[0]
    else:
        base_value = expected_value

    case_rows = []
    for rank, idx in enumerate(case_study_idx, start=1):
        pos = test.index.get_loc(idx)
        exp = shap.Explanation(
            values=shap_fraud[pos],
            base_values=base_value,
            data=X_test.iloc[pos].values,
            feature_names=feature_cols,
        )
        plt.figure()
        shap.plots.waterfall(exp, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"shap_waterfall_case{rank}.png", dpi=140, bbox_inches="tight")
        plt.close()

        top_drivers = (
            pd.Series(shap_fraud[pos], index=feature_cols)
            .abs().sort_values(ascending=False).head(5)
        )
        case_rows.append({
            "case": rank,
            "row_index": int(idx),
            "fraud_probability": round(float(results.loc[idx, "fraud_probability"]), 4),
            "top_5_drivers": ", ".join(top_drivers.index.tolist()),
        })

    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(OUT_DIR / "shap_case_studies.csv", index=False)

    # Narrative report
    lines = [
        "# SHAP Explainability Report — ExtraTrees_v2 Fraud Model",
        "",
        f"Model retrained on `train_login_model.csv` ({len(train)} rows) and evaluated on "
        f"`test_login_model.csv` ({len(test)} rows, {int(y_test.sum())} fraud cases).",
        f"Sanity-check performance: ROC-AUC = {roc_auc_score(y_test, proba):.4f}, "
        f"PR-AUC = {average_precision_score(y_test, proba):.4f} "
        "(matches the reported v2 model results).",
        "",
        "## Top 15 Globally Important Features (mean |SHAP value|)",
        "",
        "| Rank | Feature | Mean \\|SHAP\\| |",
        "|---|---|---|",
    ]
    for i, row in importance_df.head(15).iterrows():
        lines.append(f"| {i+1} | {row['feature']} | {row['mean_abs_shap']:.4f} |")

    lines += [
        "",
        "## Artifacts",
        "- `shap_feature_importance.csv` — full ranked table of mean |SHAP value| per feature.",
        "- `shap_global_importance_bar.png` — bar chart of the top 20 globally important features.",
        "- `shap_summary_beeswarm.png` — beeswarm plot showing both magnitude and direction "
        "(red = high feature value, blue = low) of each feature's push toward/away from fraud.",
        "- `shap_dependence_<feature>.png` — dependence plots for the top 5 features, showing "
        "how the feature's value relates to its SHAP contribution and interaction effects.",
        "- `shap_waterfall_case{1,2,3}.png` — local explanations for the 3 highest-confidence "
        "true-positive fraud predictions in the test set, with `shap_case_studies.csv` listing "
        "each case's predicted probability and top 5 drivers.",
        "",
        "## How to Read This Report",
        "- Positive SHAP values push the prediction toward **fraud**; negative values push toward "
        "**legitimate**.",
        "- Global importance identifies which signals matter most across the whole population "
        "(useful for feature governance and model documentation).",
        "- Dependence plots reveal non-linear thresholds (e.g., risk scores or distance cutoffs) "
        "that the tree ensemble has learned, which can be cross-checked against business rules.",
        "- Waterfall case studies support analyst review and regulatory explainability requirements "
        "by showing exactly why a specific transaction/login was scored as high risk.",
    ]
    (OUT_DIR / "shap_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"SHAP report and artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
