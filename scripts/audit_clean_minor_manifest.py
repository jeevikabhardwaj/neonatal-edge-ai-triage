from pathlib import Path
import pandas as pd


MANIFEST_PATH = Path(
    "data/splits/minor/minor_frame_manifest_clean.csv"
)


def main():
    print("Loading clean Minor manifest...")

    df = pd.read_csv(
        MANIFEST_PATH,
        dtype={"record_id": str},
        low_memory=False,
    )

    print("\n========== BASIC COUNTS ==========")
    print("Total rows:", len(df))
    print("Unique records:", df["record_id"].nunique())
    print("Unique folders:", df["folder"].nunique())

    # ---------------------------------------------------------
    # 1. RECORD-LEVEL SPLIT CONSISTENCY
    # ---------------------------------------------------------

    record_split_counts = (
        df.groupby("record_id")["split"]
        .nunique()
    )

    records_multiple_splits = (
        record_split_counts[record_split_counts > 1]
    )

    print("\n========== RECORD SPLIT CHECK ==========")
    print(
        "Records appearing in multiple splits:",
        len(records_multiple_splits)
    )

    # ---------------------------------------------------------
    # 2. RECORD-LEVEL LABEL CONSISTENCY
    # ---------------------------------------------------------

    record_label_counts = (
        df.groupby("record_id")["label"]
        .nunique()
    )

    records_multiple_labels = (
        record_label_counts[record_label_counts > 1]
    )

    print(
        "Records with conflicting labels:",
        len(records_multiple_labels)
    )

    # ---------------------------------------------------------
    # 3. PHYSICAL IMAGE KEY
    #
    # A physical image is identified by:
    # source ZIP/file + archive member/image name
    # ---------------------------------------------------------

    df["image_key"] = (
        df["source_file"].fillna("").astype(str)
        + "||"
        + df["archive_member"].fillna("").astype(str)
    )

    unique_images = df["image_key"].nunique()

    print("\n========== IMAGE KEY CHECK ==========")
    print("Unique physical image keys:", unique_images)
    print("Manifest rows:", len(df))

    duplicate_image_keys = (
        df.groupby("image_key")
        .size()
    )

    duplicate_image_keys = (
        duplicate_image_keys[duplicate_image_keys > 1]
    )

    print(
        "Duplicate physical image keys:",
        len(duplicate_image_keys)
    )

    # ---------------------------------------------------------
    # 4. IMAGES ASSIGNED TO MULTIPLE RECORDS
    # ---------------------------------------------------------

    image_record_counts = (
        df.groupby("image_key")["record_id"]
        .nunique()
    )

    images_multiple_records = (
        image_record_counts[image_record_counts > 1]
    )

    print(
        "Images assigned to multiple records:",
        len(images_multiple_records)
    )

    # ---------------------------------------------------------
    # 5. IMAGES APPEARING IN MULTIPLE SPLITS
    # ---------------------------------------------------------

    image_split_counts = (
        df.groupby("image_key")["split"]
        .nunique()
    )

    images_multiple_splits = (
        image_split_counts[image_split_counts > 1]
    )

    print(
        "Images appearing in multiple splits:",
        len(images_multiple_splits)
    )

    # ---------------------------------------------------------
    # 6. IMAGES WITH CONFLICTING LABELS
    # ---------------------------------------------------------

    image_label_counts = (
        df.groupby("image_key")["label"]
        .nunique()
    )

    images_conflicting_labels = (
        image_label_counts[image_label_counts > 1]
    )

    print(
        "Images with conflicting labels:",
        len(images_conflicting_labels)
    )

    # ---------------------------------------------------------
    # 7. FOLDER ↔ RECORD CONSISTENCY
    # ---------------------------------------------------------

    folder_record_counts = (
        df.groupby("folder")["record_id"]
        .nunique()
    )

    folders_multiple_records = (
        folder_record_counts[folder_record_counts > 1]
    )

    print("\n========== FOLDER CHECK ==========")
    print(
        "Folders assigned to multiple records:",
        len(folders_multiple_records)
    )

    # ---------------------------------------------------------
    # 8. SPLIT DISTRIBUTION
    # ---------------------------------------------------------

    print("\n========== SPLIT DISTRIBUTION ==========")
    print(
        df.groupby("split")["record_id"]
        .nunique()
    )

    # ---------------------------------------------------------
    # 9. FINAL PASS/FAIL
    # ---------------------------------------------------------

    checks = {
        "226 unique records":
            df["record_id"].nunique() == 226,

        "226 unique folders":
            df["folder"].nunique() == 226,

        "No records across multiple splits":
            len(records_multiple_splits) == 0,

        "No conflicting record labels":
            len(records_multiple_labels) == 0,

        "No duplicate physical image keys":
            len(duplicate_image_keys) == 0,

        "No image assigned to multiple records":
            len(images_multiple_records) == 0,

        "No image across multiple splits":
            len(images_multiple_splits) == 0,

        "No conflicting image labels":
            len(images_conflicting_labels) == 0,

        "No folder assigned to multiple records":
            len(folders_multiple_records) == 0,
    }

    print("\n========== FINAL INTEGRITY AUDIT ==========")

    all_pass = True

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")

        if not passed:
            all_pass = False

    print("\n==========================================")

    if all_pass:
        print("✅ INTEGRITY AUDIT PASSED")
        print("The clean Minor manifest is safe to use for training.")
    else:
        print("❌ INTEGRITY AUDIT FAILED")
        print("DO NOT TRAIN THE MODEL.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
