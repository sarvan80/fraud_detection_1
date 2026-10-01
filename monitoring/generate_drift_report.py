"""
Data / concept drift monitoring report.

Compares the distribution of every model feature (plus the target) between a
"reference" population and a "current" population, using:
  - Population Stability Index (PSI) for numeric drift magnitude
  - Kolmogorov-Smirnov test for distributional shape drift
  - Categorical chi-square test for categorical/one-hot columns
  - Fraud-rate drift (prevalence shift) as a special case

Two comparisons are produced:
  1. train_login_model.csv (reference) vs test_login_model.csv (current)
     -> the standard train/serving drift check for the canonical pipeline.
  2. data/*.csv (reference: older 31.24% fraud-rate generation) vs
     users_10000.csv/login_events_10000.csv/transactions_10000.csv (current:
     2.44% fraud-rate generation) -> a real, large drift example already
     present in this project, useful for demonstrating severe drift.

Outputs (per comparison) are written to monitoring/<name>/:
  - drift_report.csv          (per-feature PSI/KS/chi2 + drift flag)
  - drift_summary.md          (narrative report incl. top drifted features)
  - drift_psi_barplot.png     (top 20 features by PSI)
  - <feature>_distribution.png for the top 6 drifted numeric features
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

FOLDER = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
OUT_ROOT = FOLDER / "monitoring"
OUT_ROOT.mkdir(exist_ok=True)

PSI_WARN = 0.10
PSI_ALERT = 0.25


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    if expected.nunique() <= 1 or len(expected) == 0 or len(actual) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        cuts = np.linspace(expected.min(), expected.max() + 1e-9, bins + 1)
    e_counts, edges = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_perc = e_counts / max(len(expected), 1) + 1e-6
    a_perc = a_counts / max(len(actual), 1) + 1e-6
    return float(np.sum((e_perc - a_perc) * np.log(e_perc / a_perc)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    if len(cats) <= 1:
        return 0.0
    e_counts = expected.value_counts(normalize=True).reindex(cats, fill_value=0) + 1e-6
    a_counts = actual.value_counts(normalize=True).reindex(cats, fill_value=0) + 1e-6
    return float(np.sum((e_counts - a_counts) * np.log(e_counts / a_counts)))


def drift_flag(psi_value: float) -> str:
    if psi_value >= PSI_ALERT:
        return "ALERT"
    if psi_value >= PSI_WARN:
        return "WARN"
    return "OK"


def analyze(reference: pd.DataFrame, current: pd.DataFrame, name: str, target_col: str = "is_fraud"):
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    common_cols = [c for c in reference.columns if c in current.columns]
    rows = []

    for col in common_cols:
        ref_series = reference[col]
        cur_series = current[col]
        is_numeric = pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(cur_series)
        is_bool = pd.api.types.is_bool_dtype(ref_series) or set(pd.unique(ref_series.dropna())) <= {0, 1, True, False}

        if is_numeric and not is_bool and ref_series.nunique() > 5:
            psi_val = psi_numeric(ref_series, cur_series)
            try:
                ks_stat, ks_p = ks_2samp(ref_series.dropna(), cur_series.dropna())
            except Exception:
                ks_stat, ks_p = np.nan, np.nan
            test_used = "PSI + KS"
        else:
            # categorical / boolean / low-cardinality numeric
            ref_cat = ref_series.fillna("__missing__").astype(str)
            cur_cat = cur_series.fillna("__missing__").astype(str)
            psi_val = psi_categorical(ref_cat, cur_cat)
            cats = sorted(set(ref_cat.unique().tolist()) | set(cur_cat.unique().tolist()))
            try:
                cont = pd.DataFrame({
                    "ref": ref_cat.value_counts().reindex(cats, fill_value=0),
                    "cur": cur_cat.value_counts().reindex(cats, fill_value=0),
                }).T
                if cont.shape[1] >= 2 and cont.to_numpy().sum() > 0:
                    _, ks_p, _, _ = chi2_contingency(cont.to_numpy() + 1)
                    ks_stat = np.nan
                else:
                    ks_stat, ks_p = np.nan, np.nan
            except Exception:
                ks_stat, ks_p = np.nan, np.nan
            test_used = "PSI + Chi-square"

        rows.append({
            "feature": col,
            "test_used": test_used,
            "psi": round(psi_val, 4),
            "ks_or_chi2_pvalue": round(ks_p, 4) if pd.notna(ks_p) else np.nan,
            "reference_mean_or_top": round(ref_series.mean(), 4) if is_numeric else ref_series.astype(str).mode().iloc[0],
            "current_mean_or_top": round(cur_series.mean(), 4) if is_numeric else cur_series.astype(str).mode().iloc[0],
            "drift_status": drift_flag(psi_val),
        })

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    report.to_csv(out_dir / "drift_report.csv", index=False)

    # Prevalence / concept drift on the target
    prevalence_line = ""
    if target_col in common_cols:
        ref_rate = reference[target_col].mean() * 100
        cur_rate = current[target_col].mean() * 100
        rel_change = ((cur_rate - ref_rate) / ref_rate * 100) if ref_rate else float("nan")
        prevalence_line = (
            f"- **Target prevalence drift**: reference fraud rate = {ref_rate:.2f}% "
            f"vs current fraud rate = {cur_rate:.2f}% "
            f"({rel_change:+.1f}% relative change).\n"
        )

    # Plot: top 20 features by PSI
    top20 = report.head(20)
    plt.figure(figsize=(10, 8))
    colors = top20["drift_status"].map({"ALERT": "crimson", "WARN": "orange", "OK": "steelblue"})
    plt.barh(top20["feature"][::-1], top20["psi"][::-1], color=colors[::-1])
    plt.axvline(PSI_WARN, color="orange", linestyle="--", linewidth=1, label=f"WARN ({PSI_WARN})")
    plt.axvline(PSI_ALERT, color="red", linestyle="--", linewidth=1, label=f"ALERT ({PSI_ALERT})")
    plt.xlabel("Population Stability Index (PSI)")
    plt.title(f"Top Feature Drift — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "drift_psi_barplot.png", dpi=140)
    plt.close()

    # Distribution plots for top 6 drifted numeric features
    numeric_drifted = [
        r["feature"] for r in rows
        if r["test_used"] == "PSI + KS"
    ]
    numeric_drifted = report[report["feature"].isin(numeric_drifted)]["feature"].head(6).tolist()
    for feat in numeric_drifted:
        plt.figure(figsize=(6, 4))
        plt.hist(reference[feat].dropna(), bins=30, alpha=0.5, density=True, label="reference")
        plt.hist(current[feat].dropna(), bins=30, alpha=0.5, density=True, label="current")
        plt.title(f"Distribution shift: {feat}")
        plt.legend()
        plt.tight_layout()
        safe_name = feat.replace("/", "_")
        plt.savefig(out_dir / f"{safe_name}_distribution.png", dpi=140)
        plt.close()

    n_alert = (report["drift_status"] == "ALERT").sum()
    n_warn = (report["drift_status"] == "WARN").sum()
    n_ok = (report["drift_status"] == "OK").sum()

    summary_lines = [
        f"# Drift Monitoring Report — {name}",
        "",
        f"Reference rows: {len(reference)} | Current rows: {len(current)} | Features compared: {len(report)}",
        "",
        f"- 🔴 ALERT (PSI ≥ {PSI_ALERT}): {n_alert} features",
        f"- 🟠 WARN (PSI ≥ {PSI_WARN}): {n_warn} features",
        f"- 🟢 OK: {n_ok} features",
        "",
        prevalence_line,
        "## Top 10 Most Drifted Features",
        "",
        "| Feature | Test | PSI | p-value | Reference | Current | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in report.head(10).iterrows():
        summary_lines.append(
            f"| {r['feature']} | {r['test_used']} | {r['psi']} | {r['ks_or_chi2_pvalue']} "
            f"| {r['reference_mean_or_top']} | {r['current_mean_or_top']} | {r['drift_status']} |"
        )
    summary_lines += [
        "",
        "## Interpretation Guide",
        "- **PSI < 0.10**: no significant population shift, no action needed.",
        "- **0.10 ≤ PSI < 0.25**: moderate shift — monitor and investigate root cause.",
        "- **PSI ≥ 0.25**: severe shift — retrain/recalibrate the model before trusting its scores.",
        "- KS-test p-value < 0.05 (numeric) or chi-square p-value < 0.05 (categorical) confirms the",
        "  distributions are statistically significantly different, corroborating the PSI signal.",
        "",
        "See `drift_report.csv` for the full per-feature table and `drift_psi_barplot.png` for a",
        "ranked visual summary. Feature-level distribution plots are provided for the most drifted",
        "numeric features.",
    ]
    (out_dir / "drift_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[{name}] ALERT={n_alert} WARN={n_warn} OK={n_ok} -> {out_dir}")
    return report


def main():
    # Comparison 1: canonical train vs test split (expected: low drift, sanity check)
    train = pd.read_csv(FOLDER / "train_login_model.csv")
    test = pd.read_csv(FOLDER / "test_login_model.csv")
    analyze(train, test, name="train_vs_test_canonical")

    # Comparison 2: older data/ generation (31.24% fraud) vs current root-level
    # generation (2.44% fraud) -> demonstrates a real, severe drift scenario
    # already present across this project's dataset versions.
    old_users = pd.read_csv(FOLDER / "data" / "users_10000.csv")
    old_logins = pd.read_csv(FOLDER / "data" / "login_events_10000.csv")
    old_merged = old_logins.merge(old_users, on="user_id", how="left", suffixes=("", "_user"))

    new_users = pd.read_csv(FOLDER / "users_10000.csv")
    new_logins = pd.read_csv(FOLDER / "login_events_10000.csv")
    new_merged = new_logins.merge(new_users, on="user_id", how="left", suffixes=("", "_user"))

    analyze(old_merged, new_merged, name="data_folder_vs_root_generation")


if __name__ == "__main__":
    main()
