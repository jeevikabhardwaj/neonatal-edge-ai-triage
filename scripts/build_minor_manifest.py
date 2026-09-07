from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_INDEX = PROJECT_ROOT / "openpocus_image_index.csv"
RECORD_SPLITS = PROJECT_ROOT / "openpocus_record_splits.csv"
OUTPUT = PROJECT_ROOT / "data" / "splits" / "minor_frame_manifest.csv"
BACKUP = PROJECT_ROOT / "data" / "splits" / "minor_frame_manifest_bad_before_rebuild.csv"


def main():
    print("=" * 80)
    print("BUILDING CLEAN MINOR FRAME MANIFEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load authoritative tables
    # ------------------------------------------------------------------
    splits = pd.read_csv(
        RECORD_SPLITS,
        dtype=str
    )

    images = pd.read_csv(
        IMAGE_INDEX,
        dtype=str,
        low_memory=False
    )

    print(f"Record split rows: {len(splits)}")
    print(f"Image index rows:  {len(images)}")

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------
    required_split_cols = {
        "AI ID",
        "label",
        "split",
    }

    required_image_cols = {
        "record_id",
        "folder",
        "storage_type",
        "source_file",
        "archive_member",
        "image_name",
        "zone",
        "frame",
        "zone_label",
        "normal_abnormal",
        "covid",
        "device",
    }

    missing_split = required_split_cols - set(splits.columns)
    missing_image = required_image_cols - set(images.columns)

    if missing_split:
        raise ValueError(
            f"Missing columns from record split table: {sorted(missing_split)}"
        )

    if missing_image:
        raise ValueError(
            f"Missing columns from image index: {sorted(missing_image)}"
        )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Use EXACT AI ID -> record_id.
    #
    # NEVER join on normalized_id.
    # ------------------------------------------------------------------
    splits["AI ID"] = splits["AI ID"].astype(str).str.strip()
    images["record_id"] = images["record_id"].astype(str).str.strip()

    split_ids = set(splits["AI ID"])
    image_ids = set(images["record_id"])

    print("\nExact ID validation")
    print(f"Split IDs:       {len(split_ids)}")
    print(f"Image record IDs:{len(image_ids)}")
    print(f"Exact overlap:   {len(split_ids & image_ids)}")

    missing_from_images = split_ids - image_ids
    missing_from_splits = image_ids - split_ids

    if missing_from_images:
        raise ValueError(
            "These split-table IDs are missing from image index:\n"
            + "\n".join(sorted(missing_from_images))
        )

    if missing_from_splits:
        raise ValueError(
            "These image-index IDs are missing from split table:\n"
            + "\n".join(sorted(missing_from_splits))
        )

    # ------------------------------------------------------------------
    # Make sure each record has exactly one split + label definition.
    # ------------------------------------------------------------------
    split_meta = splits[
        ["AI ID", "label", "split"]
    ].drop_duplicates()

    duplicate_split_definitions = (
        split_meta
        .groupby("AI ID")
        .size()
    )

    bad_split_definitions = duplicate_split_definitions[
        duplicate_split_definitions > 1
    ]

    if len(bad_split_definitions):
        print("\nERROR: Multiple split definitions detected:")
        print(bad_split_definitions)
        raise ValueError("Record split table is not one-row-per-record.")

    # ------------------------------------------------------------------
    # Exact merge.
    #
    # AI ID == record_id
    # ------------------------------------------------------------------
    manifest = images.merge(
        split_meta,
        left_on="record_id",
        right_on="AI ID",
        how="inner",
        validate="many_to_one",
    )

    # Remove duplicate join column.
    manifest = manifest.drop(columns=["AI ID"])

    # ------------------------------------------------------------------
    # Validate record coverage.
    # ------------------------------------------------------------------
    manifest_record_ids = set(
        manifest["record_id"].astype(str)
    )

    if manifest_record_ids != image_ids:
        raise ValueError(
            "Manifest record coverage does not match image index."
        )

    if manifest_record_ids != split_ids:
        raise ValueError(
            "Manifest record coverage does not match split table."
        )

    print("\nRecord coverage:")
    print(f"Unique manifest records: {manifest['record_id'].nunique()}")
    print(f"Expected records:        {len(split_ids)}")

    # ------------------------------------------------------------------
    # Backup existing manifest before replacing it.
    # ------------------------------------------------------------------
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        OUTPUT.replace(BACKUP)
        print(f"\nOld manifest backed up to:")
        print(BACKUP)

    # ------------------------------------------------------------------
    # Save clean manifest.
    # ------------------------------------------------------------------
    manifest.to_csv(
        OUTPUT,
        index=False
    )

    print("\nClean manifest written:")
    print(OUTPUT)

    # ------------------------------------------------------------------
    # Basic summary
    # ------------------------------------------------------------------
    print("\nSplit distribution:")
    print(
        manifest.groupby("split")["record_id"]
        .nunique()
        .to_string()
    )

    print("\nLabel distribution:")
    print(
        manifest.groupby("label")["record_id"]
        .nunique()
        .to_string()
    )

    print("\nRows:")
    print(len(manifest))

    print("\nColumns:")
    print(manifest.columns.tolist())

    print("\n" + "=" * 80)
    print("MANIFEST REBUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
