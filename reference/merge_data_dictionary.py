"""
Merge the original data dictionary (data_dictionary.csv) with the three
feature catalogs (device, identity/account, networking) transcribed from
the Transmit-SDK-style PNGs, producing a single consolidated
data_dictionary_new.csv.

A new boolean-style column `is_new_field` marks every row that was added
in this update (i.e. every row sourced from the three feature catalogs)
as "Yes", while all originally-documented attributes keep "No". For rows
coming from the networking catalog specifically, the `catalog_new_flag`
column preserves the finer-grained "New Feature (Beyond Existing
Catalogs)" tag (Yes/No) that was assigned when that catalog was created,
since some networking features were flagged as overlapping with concepts
already present in the device/identity catalogs.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(r"c:\Users\thiag\Downloads\Schwab-Fraud detection")
REF_DIR = ROOT / "reference"
EDA_DIR = ROOT / "data_exploraton"

ORIGINAL_DICT = REF_DIR / "data_dictionary.csv"
DEVICE_CATALOG = EDA_DIR / "device_feature_catalog.csv"
IDENTITY_CATALOG = EDA_DIR / "identity_account_features_catalog.csv"
NETWORK_CATALOG = EDA_DIR / "networking_feature_catalog.csv"

OUTPUT_CSV = REF_DIR / "data_dictionary_new.csv"

UNIFIED_COLUMNS = [
    "dataset",
    "attribute",
    "data_type",
    "description",
    "example_values",
    "source_or_derivation",
    "is_new_field",
    "catalog_new_flag",
]


def load_original():
    df = pd.read_csv(ORIGINAL_DICT)
    df["is_new_field"] = "No"
    df["catalog_new_flag"] = ""
    return df[UNIFIED_COLUMNS]


def catalog_to_unified(csv_path: Path, dataset_label: str, has_new_flag_col: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    def build_description(row):
        parts = [f"Feature group: {row['Feature group']}"]
        parts.append(f"Derivation logic: {row['Pseudo derivation logic']}")
        return " | ".join(parts)

    def build_examples(row):
        return (
            f"ATO importance: {row['ATO importance']}; "
            f"ATO scenario(s): {row['ATO scenario(s) supported']}; "
            f"Available in Transmit DRS: {row['Attribute readily available in Transmit DRS']}"
        )

    def build_source(row):
        return (
            f"Raw attributes: {row['Raw Attributes available in Transmit SDK']}; "
            f"Derived from/lineage: {row['Derived from / lineage']}"
        )

    unified = pd.DataFrame({
        "dataset": dataset_label,
        "attribute": df["Feature"],
        "data_type": df["Type (Raw / Derived / Composite Derived)"],
        "description": df.apply(build_description, axis=1),
        "example_values": df.apply(build_examples, axis=1),
        "source_or_derivation": df.apply(build_source, axis=1),
        "is_new_field": "Yes",
    })

    if has_new_flag_col:
        unified["catalog_new_flag"] = df["New Feature (Beyond Existing Catalogs)"]
    else:
        unified["catalog_new_flag"] = ""

    return unified[UNIFIED_COLUMNS]


def main():
    original = load_original()
    device = catalog_to_unified(DEVICE_CATALOG, "Device Feature Catalog", has_new_flag_col=False)
    identity = catalog_to_unified(IDENTITY_CATALOG, "Identity & Account Feature Catalog", has_new_flag_col=False)
    networking = catalog_to_unified(NETWORK_CATALOG, "Networking Feature Catalog", has_new_flag_col=True)

    combined = pd.concat([original, device, identity, networking], ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {OUTPUT_CSV} with {len(combined)} total rows.")
    print(f"  Original dictionary rows (is_new_field=No): {len(original)}")
    print(f"  Device catalog rows added: {len(device)}")
    print(f"  Identity & account catalog rows added: {len(identity)}")
    print(f"  Networking catalog rows added: {len(networking)}")
    new_count = (combined["is_new_field"] == "Yes").sum()
    print(f"  Total rows tagged is_new_field=Yes: {new_count}")
    overlap_count = (combined["catalog_new_flag"] == "No").sum()
    print(f"  Networking rows tagged catalog_new_flag=No (overlap with device/identity): {overlap_count}")


if __name__ == "__main__":
    main()
