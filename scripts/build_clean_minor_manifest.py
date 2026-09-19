from pathlib import Path
import pandas as pd


INDEX_PATH = Path("openpocus_image_index_clean.csv")
SPLITS_PATH = Path("openpocus_record_splits.csv")
OUTPUT_PATH = Path("data/splits/minor/minor_frame_manifest_clean.csv")


def normalize_id(value):
    return str(value).strip()


def main():
    print("Loading clean image index...")
    index = pd.read_csv(
        INDEX_PATH,
        dtype={"record_id": str},
        low_memory=False,
    )

    print("Loading record-level splits...")
    splits = pd.read_csv(
        SPLITS_PATH,
        dtype={"AI ID": str},
        low_memory=False,
    )

    # Normalize record IDs without changing their actual meaning.
    index["record_id"] = index["record_id"].map(normalize_id)
    splits["record_id"] = splits["AI ID"].map(normalize_id)

    # Basic checks before merging.
    index_ids = set(index["record_id"])
    split_ids = set(splits["record_id"])

    if index_ids != split_ids:
        print("ERROR: record IDs do not match.")
        print("Only in image index:", sorted(index_ids - split_ids))
        print("Only in splits:", sorted(split_ids - index_ids))
        raise SystemExit(1)

    if len(index_ids) != 226:
        raise ValueError(
            f"Expected 226 unique records, found {len(index_ids)}"
        )

    if len(splits) != 226:
        raise ValueError(
            f"Expected 226 split records, found {len(splits)}"
        )

    # Keep only the record-level information needed by the Minor experiment.
    split_info = splits[
        [
            "record_id",
            "label",
            "split",
            "Normal or Not (0=normal; 1=abnormal)",
            "COVID +, 0=no, 1=yes",
            "Frames",
        ]
    ].copy()

    # Ensure every record has exactly one split/label assignment.
    if split_info["record_id"].duplicated().any():
        dupes = split_info.loc[
            split_info["record_id"].duplicated(keep=False),
            "record_id",
        ].tolist()
        raise ValueError(
            f"Duplicate record assignments found: {dupes}"
        )

    # Merge image-level information with record-level split/label.
    manifest = index.merge(
        split_info,
        on="record_id",
        how="left",
        validate="many_to_one",
    )

    # Every image must receive a split and label.
    if manifest["split"].isna().any():
        raise ValueError("Some images have no split assignment.")

    if manifest["label"].isna().any():
        raise ValueError("Some images have no label.")

    # Keep a clean, explicit column order.
    columns = [
        "record_id",
        "folder",
        "storage_type",
        "source_file",
        "archive_member",
        "image_name",
        "zone",
        "frame",
        "normal_abnormal",
        "covid",
        "device",
        "label",
        "split",
        "Frames",
    ]

    manifest = manifest[columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)

    print("\n========== CLEAN MINOR MANIFEST ==========")
    print("Output:", OUTPUT_PATH)
    print("Rows:", len(manifest))
    print("Unique records:", manifest["record_id"].nunique())
    print("Unique folders:", manifest["folder"].nunique())

    print("\nSplit distribution:")
    print(manifest.groupby("split")["record_id"].nunique())

    print("\nImage distribution by split:")
    print(manifest["split"].value_counts())

    print("\nLabel distribution by record:")
    print(
        manifest[
            ["record_id", "label", "split"]
        ]
        .drop_duplicates()
        .groupby(["split", "label"])
        .size()
    )

    print("\nManifest created successfully.")


if __name__ == "__main__":
    main()
