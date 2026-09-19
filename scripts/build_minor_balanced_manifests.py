from pathlib import Path
import pandas as pd
import numpy as np


SOURCE = Path(
    "data/splits/minor/minor_frame_manifest_clean.csv"
)

OUTPUT_DIR = Path("data/splits/minor")

TRAIN_OUTPUT = OUTPUT_DIR / "minor_train_manifest_240.csv"
VAL_OUTPUT = OUTPUT_DIR / "minor_validation_manifest_240.csv"

FRAMES_PER_RECORD = 240


def sample_record(group):
    """
    Deterministically select exactly 240 frames from one record.

    The dataframe is first sorted by zone and frame so that the
    same input always produces the same selected frames.
    """
    group = group.sort_values(
        ["zone", "frame", "image_name"]
    ).reset_index(drop=True)

    if len(group) < FRAMES_PER_RECORD:
        raise ValueError(
            f"Record {group.iloc[0]['record_id']} has only "
            f"{len(group)} frames; cannot sample {FRAMES_PER_RECORD}."
        )

    # Uniformly distribute the selected frames across the
    # complete ordered record.
    indices = np.linspace(
        0,
        len(group) - 1,
        FRAMES_PER_RECORD,
        dtype=int,
    )

    # np.linspace + dtype=int can theoretically produce
    # duplicate indices for smaller arrays. Verify explicitly.
    indices = np.unique(indices)

    if len(indices) != FRAMES_PER_RECORD:
        raise ValueError(
            f"Sampling produced {len(indices)} unique frames "
            f"instead of {FRAMES_PER_RECORD} for record "
            f"{group.iloc[0]['record_id']}."
        )

    return group.iloc[indices].copy()


def validate_sampled_manifest(df, expected_records):
    print("\n========== VALIDATING DERIVED MANIFEST ==========")

    record_counts = (
        df.groupby("record_id")
        .size()
    )

    if len(record_counts) != expected_records:
        raise ValueError(
            f"Expected {expected_records} records, "
            f"found {len(record_counts)}."
        )

    wrong_counts = record_counts[
        record_counts != FRAMES_PER_RECORD
    ]

    if len(wrong_counts) > 0:
        print(wrong_counts)
        raise ValueError(
            "Some records do not contain exactly "
            f"{FRAMES_PER_RECORD} frames."
        )

    duplicate_keys = (
        df.groupby("image_key")
        .size()
    )

    duplicate_keys = duplicate_keys[
        duplicate_keys > 1
    ]

    if len(duplicate_keys) > 0:
        raise ValueError(
            f"Found {len(duplicate_keys)} duplicate "
            "physical image keys."
        )

    print(
        f"Records: {len(record_counts)}"
    )
    print(
        f"Images: {len(df)}"
    )
    print(
        f"Images per record: "
        f"{record_counts.min()}–{record_counts.max()}"
    )
    print(
        "Duplicate physical images: 0"
    )

    print("\nSplit:")
    print(
        df.groupby("split")["record_id"]
        .nunique()
    )

    print("\nLabel:")
    print(
        df[
            ["record_id", "label", "split"]
        ]
        .drop_duplicates()
        .groupby(["split", "label"])
        .size()
    )


def main():

    print("Loading clean manifest...")

    df = pd.read_csv(
        SOURCE,
        dtype={"record_id": str},
        low_memory=False,
    )

    print(
        f"Loaded {len(df):,} rows."
    )

    # ---------------------------------------------------------
    # Create a physical image key.
    # ---------------------------------------------------------

    df["image_key"] = (
        df["source_file"].fillna("").astype(str)
        + "||"
        + df["archive_member"].fillna("").astype(str)
    )

    if df["image_key"].duplicated().any():
        raise ValueError(
            "The source manifest already contains duplicate "
            "physical image keys."
        )

    # ---------------------------------------------------------
    # Confirm the source manifest is exactly what we expect.
    # ---------------------------------------------------------

    if df["record_id"].nunique() != 226:
        raise ValueError(
            "Expected 226 source records."
        )

    train_records = (
        df[df["split"] == "train"]
        ["record_id"]
        .nunique()
    )

    val_records = (
        df[df["split"] == "validation"]
        ["record_id"]
        .nunique()
    )

    test_records = (
        df[df["split"] == "test"]
        ["record_id"]
        .nunique()
    )

    print("\nSource record counts:")
    print("Train:", train_records)
    print("Validation:", val_records)
    print("Test:", test_records)

    # ---------------------------------------------------------
    # Sample TRAIN and VALIDATION only.
    # ---------------------------------------------------------

    derived = []

    for split in ["train", "validation"]:

        split_df = df[
            df["split"] == split
        ].copy()

        print(
            f"\nSampling {split}..."
        )

        sampled_groups = []

        for record_id, group in split_df.groupby(
            "record_id",
            sort=True
        ):

            sampled = sample_record(group)

            sampled_groups.append(sampled)

        sampled_split = pd.concat(
            sampled_groups,
            ignore_index=True,
        )

        derived.append(sampled_split)

    train_sampled = derived[0]
    val_sampled = derived[1]

    # ---------------------------------------------------------
    # Ensure no sampled image overlaps.
    # ---------------------------------------------------------

    train_keys = set(
        train_sampled["image_key"]
    )

    val_keys = set(
        val_sampled["image_key"]
    )

    overlap = train_keys & val_keys

    if overlap:
        raise ValueError(
            f"Train/validation image overlap detected: "
            f"{len(overlap)}"
        )

    # ---------------------------------------------------------
    # Ensure sampled data contains ONLY its intended split.
    # ---------------------------------------------------------

    if set(train_sampled["split"]) != {"train"}:
        raise ValueError(
            "Training manifest contains non-training records."
        )

    if set(val_sampled["split"]) != {"validation"}:
        raise ValueError(
            "Validation manifest contains non-validation records."
        )

    # ---------------------------------------------------------
    # Drop temporary image key before saving.
    # ---------------------------------------------------------

    train_sampled = train_sampled.drop(
        columns=["image_key"]
    )

    val_sampled = val_sampled.drop(
        columns=["image_key"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_sampled.to_csv(
        TRAIN_OUTPUT,
        index=False,
    )

    val_sampled.to_csv(
        VAL_OUTPUT,
        index=False,
    )

    # ---------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------

    train_check = pd.read_csv(
        TRAIN_OUTPUT,
        dtype={"record_id": str},
        low_memory=False,
    )

    val_check = pd.read_csv(
        VAL_OUTPUT,
        dtype={"record_id": str},
        low_memory=False,
    )

    train_check["image_key"] = (
        train_check["source_file"].fillna("").astype(str)
        + "||"
        + train_check["archive_member"].fillna("").astype(str)
    )

    val_check["image_key"] = (
        val_check["source_file"].fillna("").astype(str)
        + "||"
        + val_check["archive_member"].fillna("").astype(str)
    )

    validate_sampled_manifest(
        train_check,
        train_records,
    )

    validate_sampled_manifest(
        val_check,
        val_records,
    )

    print("\n========== FINAL SUMMARY ==========")

    print(
        f"Training: "
        f"{len(train_check):,} images "
        f"from {train_records} records"
    )

    print(
        f"Validation: "
        f"{len(val_check):,} images "
        f"from {val_records} records"
    )

    print(
        f"Test remains untouched: "
        f"{len(df[df['split'] == 'test']):,} images "
        f"from {test_records} records"
    )

    print("\nOutput files:")
    print(TRAIN_OUTPUT)
    print(VAL_OUTPUT)

    print(
        "\n✅ RECORD-BALANCED MANIFESTS CREATED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
