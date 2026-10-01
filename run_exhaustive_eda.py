from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "data_exploraton"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

DATASETS = {
    "users": DATA_DIR / "users_10000.csv",
    "login_events": DATA_DIR / "login_events_10000.csv",
    "transactions": DATA_DIR / "transactions_10000.csv",
}


def save_plot(name):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close()


def safe_rate(series):
    return float(series.mean()) if len(series) else None


def dataframe_markdown(df):
    if df.empty:
        return "_No rows._"
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for values in df.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def profile_dataframe(name, df):
    rows = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        row = {
            "dataset": name,
            "column": col,
            "dtype": str(series.dtype),
            "rows": int(len(df)),
            "missing_count": missing,
            "missing_pct": round(100 * missing / max(len(df), 1), 4),
            "unique_count": unique,
            "unique_pct": round(100 * unique / max(len(df), 1), 4),
            "duplicate_value_count": int(series.duplicated(keep=False).sum()),
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            row.update({
                "min": numeric_series.min(),
                "max": numeric_series.max(),
                "mean": numeric_series.mean(),
                "median": numeric_series.median(),
                "std": numeric_series.std(),
                "q01": numeric_series.quantile(0.01),
                "q25": numeric_series.quantile(0.25),
                "q75": numeric_series.quantile(0.75),
                "q99": numeric_series.quantile(0.99),
            })
        else:
            row.update({
                "min": None, "max": None, "mean": None, "median": None,
                "std": None, "q01": None, "q25": None, "q75": None, "q99": None,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_counts(df, dataset, max_values=50):
    frames = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            counts = df[col].fillna("<MISSING>").astype(str).value_counts().head(max_values)
            part = counts.rename("count").reset_index()
            part.columns = ["value", "count"]
            part.insert(0, "column", col)
            part.insert(0, "dataset", dataset)
            part["pct"] = 100 * part["count"] / max(len(df), 1)
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def numeric_summary(df, dataset):
    numeric = df.select_dtypes(include=[np.number]).select_dtypes(exclude=["bool"])
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.reset_index()
    summary = summary.rename(columns={"index": "column"})
    summary.insert(0, "dataset", dataset)
    summary["missing_count"] = numeric.isna().sum().values
    summary["zero_count"] = (numeric == 0).sum().values
    return summary


def fraud_breakdown(df, dataset):
    if "is_fraud" not in df.columns:
        return pd.DataFrame()
    fraud = df["is_fraud"].fillna(0).astype(int)
    result = pd.DataFrame({
        "dataset": [dataset],
        "rows": [len(df)],
        "fraud_rows": [int(fraud.sum())],
        "legitimate_rows": [int((fraud == 0).sum())],
        "fraud_rate_pct": [100 * fraud.mean()],
    })
    if "fraud_type" in df.columns:
        type_counts = df.loc[fraud == 1, "fraud_type"].fillna("unknown").value_counts()
        for value, count in type_counts.items():
            result[f"fraud_type__{value}"] = int(count)
    return result


def group_fraud_rates(df, dataset, column, top_n=25):
    if "is_fraud" not in df.columns or column not in df.columns:
        return pd.DataFrame()
    working = df[[column, "is_fraud"]].copy()
    working[column] = working[column].fillna("<MISSING>").astype(str)
    grouped = working.groupby(column)["is_fraud"].agg(["count", "sum", "mean"]).reset_index()
    grouped["dataset"] = dataset
    grouped["fraud_rate_pct"] = grouped["mean"] * 100
    grouped = grouped.rename(columns={"sum": "fraud_count"})
    return grouped.sort_values(["count", "fraud_rate_pct"], ascending=False).head(top_n)


def outlier_summary(df, dataset):
    rows = []
    for col in df.select_dtypes(include=[np.number]).select_dtypes(exclude=["bool"]).columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        rows.append({
            "dataset": dataset,
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(outliers),
            "outlier_pct": 100 * outliers / len(series),
        })
    return pd.DataFrame(rows)


def create_numeric_histograms(df, dataset):
    numeric_cols = df.select_dtypes(include=[np.number]).select_dtypes(exclude=["bool"]).columns.tolist()
    for col in numeric_cols:
        values = df[col].dropna()
        if values.empty or values.nunique() <= 1:
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=30, color="#244a73", alpha=0.85)
        plt.title(f"{dataset}: distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        save_plot(f"{dataset}__hist__{col}")


def create_categorical_plots(df, dataset):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            continue
        counts = df[col].fillna("<MISSING>").astype(str).value_counts().head(15)
        if len(counts) < 2:
            continue
        plt.figure(figsize=(8, 4.5))
        counts.sort_values().plot(kind="barh", color="#e8752a")
        plt.title(f"{dataset}: top values for {col}")
        plt.xlabel("count")
        plt.ylabel(col)
        save_plot(f"{dataset}__bar__{col}")


def create_fraud_plots(df, dataset):
    if "is_fraud" not in df.columns:
        return
    fraud = df["is_fraud"].fillna(0).astype(int)
    plt.figure(figsize=(5, 4))
    fraud.value_counts().sort_index().plot(kind="bar", color=["#2b6f9e", "#d9534f"])
    plt.xticks([0, 1], ["legitimate", "fraud"], rotation=0)
    plt.ylabel("count")
    plt.title(f"{dataset}: target balance")
    save_plot(f"{dataset}__fraud_balance")

    if "fraud_type" in df.columns:
        counts = df.loc[fraud == 1, "fraud_type"].fillna("unknown").value_counts()
        if not counts.empty:
            plt.figure(figsize=(8, 4.5))
            counts.sort_values().plot(kind="barh", color="#d9534f")
            plt.title(f"{dataset}: fraud types")
            plt.xlabel("count")
            save_plot(f"{dataset}__fraud_types")

    numeric = df.select_dtypes(include=[np.number]).select_dtypes(exclude=["bool"]).columns
    for col in numeric:
        if col == "is_fraud" or df[col].nunique() < 2:
            continue
        plt.figure(figsize=(6, 4))
        df.boxplot(column=col, by="is_fraud", grid=False)
        plt.suptitle("")
        plt.title(f"{dataset}: {col} by fraud label")
        plt.xlabel("is_fraud")
        plt.ylabel(col)
        save_plot(f"{dataset}__fraud_box__{col}")


def create_correlation_plot(df, dataset):
    numeric = df.select_dtypes(include=[np.number]).select_dtypes(exclude=["bool"])
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr(numeric_only=True)
    corr.to_csv(TABLE_DIR / f"{dataset}__correlation_matrix.csv")
    plt.figure(figsize=(max(8, 0.45 * len(corr.columns)), max(6, 0.45 * len(corr.columns))))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title(f"{dataset}: numeric correlation matrix")
    save_plot(f"{dataset}__correlation_heatmap")


def write_report(summary, dataset_profiles, fraud_summary, join_summary, top_findings):
    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "Generated from the CSV files under `data`.",
        "",
        "## Scope",
        "",
        "This analysis covers file inventories, schema and quality profiling, duplicate and missing-value checks, numeric distributions, categorical frequencies, outlier screening, correlations, fraud balance, fraud-type composition, group-level fraud rates, temporal patterns, and cross-dataset consistency checks.",
        "",
        "## Dataset inventory",
        "",
        "| Dataset | Rows | Columns | Duplicate rows | Fraud rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['rows']:,} | {row['columns']} | "
            f"{row['duplicate_rows']:,} | {row.get('fraud_rate_pct', 'n/a')}% |"
        )
    lines += ["", "## Key findings", ""]
    lines.extend([f"- {finding}" for finding in top_findings])
    lines += ["", "## Fraud summary", "", dataframe_markdown(fraud_summary), "", "## Cross-dataset checks", "", dataframe_markdown(join_summary)]
    lines += [
        "",
        "## Output directories",
        "",
        "- `tables`: schema, quality, summaries, group rates, correlations, and temporal tables.",
        "- `plots`: distributions, target balance, fraud-type charts, fraud-vs-feature boxplots, and correlation heatmaps.",
        "- `eda_summary.json`: machine-readable run summary.",
        "",
        "## Interpretation notes",
        "",
        "- Fraud rates are descriptive and reflect the synthetic labels supplied in the input files.",
        "- Transaction-level or post-outcome columns should be reviewed for target leakage before modeling.",
        "- Group-level rates with very small denominators should not be treated as stable estimates.",
        "- The IQR outlier report is a screening tool, not proof that a record is invalid.",
    ]
    (OUT_DIR / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    frames = {name: pd.read_csv(path) for name, path in DATASETS.items()}
    for name, df in frames.items():
        for col in df.columns:
            if "time" in col or col.endswith("_date"):
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed

    summary = []
    all_profiles = []
    all_numeric = []
    all_categories = []
    all_outliers = []
    all_fraud = []
    top_findings = []

    for name, df in frames.items():
        profiles = profile_dataframe(name, df)
        profiles.to_csv(TABLE_DIR / f"{name}__schema_quality.csv", index=False)
        all_profiles.append(profiles)

        numeric = numeric_summary(df, name)
        numeric.to_csv(TABLE_DIR / f"{name}__numeric_summary.csv", index=False)
        all_numeric.append(numeric)

        categories = categorical_counts(df, name)
        categories.to_csv(TABLE_DIR / f"{name}__categorical_counts.csv", index=False)
        all_categories.append(categories)

        outliers = outlier_summary(df, name)
        outliers.to_csv(TABLE_DIR / f"{name}__outliers_iqr.csv", index=False)
        all_outliers.append(outliers)

        fraud = fraud_breakdown(df, name)
        if not fraud.empty:
            all_fraud.append(fraud)
            fraud.to_csv(TABLE_DIR / f"{name}__fraud_summary.csv", index=False)
            rate = float(fraud.iloc[0]["fraud_rate_pct"])
            top_findings.append(f"{name} contains {rate:.2f}% fraud labels.")

        duplicate_rows = int(df.duplicated().sum())
        fraud_rate = round(100 * df["is_fraud"].mean(), 4) if "is_fraud" in df.columns else None
        summary.append({
            "dataset": name,
            "file": DATASETS[name].name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "duplicate_rows": duplicate_rows,
            "fraud_rate_pct": fraud_rate,
        })

        create_numeric_histograms(df, name)
        create_categorical_plots(df, name)
        create_fraud_plots(df, name)
        create_correlation_plot(df, name)

        if "is_fraud" in df.columns:
            for col in ["fraud_type", "login_result", "ip_country", "ip_region", "browser_name", "os_name",
                        "new_device_flag", "is_proxy", "is_vpn", "impossible_travel_flag", "mfa_approved"]:
                rates = group_fraud_rates(df, name, col)
                if not rates.empty:
                    rates.to_csv(TABLE_DIR / f"{name}__fraud_rate_by__{col}.csv", index=False)

            if "event_time" in df.columns:
                time_df = df[["event_time", "is_fraud"]].copy()
                time_df["date"] = pd.to_datetime(time_df["event_time"], errors="coerce").dt.date
                temporal = time_df.dropna(subset=["date"]).groupby("date").agg(
                    events=("is_fraud", "size"),
                    fraud_count=("is_fraud", "sum"),
                    fraud_rate=("is_fraud", "mean"),
                ).reset_index()
                temporal["fraud_rate_pct"] = temporal["fraud_rate"] * 100
                temporal.to_csv(TABLE_DIR / f"{name}__daily_fraud_trend.csv", index=False)
                if not temporal.empty:
                    plt.figure(figsize=(10, 4))
                    plt.plot(pd.to_datetime(temporal["date"]), temporal["fraud_rate_pct"], color="#d9534f")
                    plt.title(f"{name}: daily fraud rate")
                    plt.xlabel("date")
                    plt.ylabel("fraud rate (%)")
                    plt.grid(alpha=0.25)
                    save_plot(f"{name}__daily_fraud_rate")

    # Cross-dataset consistency checks.
    checks = []
    user_ids = set(frames["users"]["user_id"].astype(str))
    login_ids = set(frames["login_events"]["user_id"].astype(str))
    txn_ids = set(frames["transactions"]["user_id"].astype(str))
    checks.extend([
        {"check": "login user_ids present in users", "value": len(login_ids - user_ids), "status": "PASS" if not (login_ids - user_ids) else "FAIL"},
        {"check": "transaction user_ids present in users", "value": len(txn_ids - user_ids), "status": "PASS" if not (txn_ids - user_ids) else "FAIL"},
        {"check": "login user_id duplicates in users", "value": int(frames["users"]["user_id"].duplicated().sum()), "status": "PASS" if frames["users"]["user_id"].is_unique else "FAIL"},
        {"check": "login rows matching transaction rows by position", "value": int((frames["login_events"]["user_id"].values == frames["transactions"]["user_id"].values).sum()), "status": "INFO"},
        {"check": "fraud label agreement login/transaction by position", "value": int((frames["login_events"]["is_fraud"].values == frames["transactions"]["is_fraud"].values).sum()), "status": "INFO"},
    ])
    join_summary = pd.DataFrame(checks)
    join_summary.to_csv(TABLE_DIR / "cross_dataset_consistency.csv", index=False)

    # Combined tables and machine-readable summary.
    pd.concat(all_profiles, ignore_index=True).to_csv(OUT_DIR / "all_schema_quality.csv", index=False)
    pd.concat(all_numeric, ignore_index=True).to_csv(OUT_DIR / "all_numeric_summary.csv", index=False)
    pd.concat(all_categories, ignore_index=True).to_csv(OUT_DIR / "all_categorical_counts.csv", index=False)
    pd.concat(all_outliers, ignore_index=True).to_csv(OUT_DIR / "all_outliers_iqr.csv", index=False)
    fraud_summary = pd.concat(all_fraud, ignore_index=True) if all_fraud else pd.DataFrame()
    if fraud_summary.empty:
        fraud_summary = pd.DataFrame(columns=["dataset", "rows", "fraud_rows", "legitimate_rows", "fraud_rate_pct"])

    payload = {
        "input_directory": str(DATA_DIR),
        "output_directory": str(OUT_DIR),
        "datasets": summary,
        "cross_dataset_checks": checks,
        "plot_count": len(list(PLOT_DIR.glob("*.png"))),
        "table_count": len(list(TABLE_DIR.glob("*.csv"))),
    }
    (OUT_DIR / "eda_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_report(summary, frames, fraud_summary, join_summary, top_findings)
    print(f"Completed EDA for {len(frames)} datasets.")
    print(f"Created {len(list(TABLE_DIR.glob('*.csv')))} tables and {len(list(PLOT_DIR.glob('*.png')))} plots.")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
