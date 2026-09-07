from pathlib import Path
from decimal import Decimal, InvalidOperation
import re
import zipfile
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/Users/D/Downloads/Lung Database")

RECORD_SPLITS = PROJECT_ROOT / "openpocus_record_splits.csv"

OUTPUT = PROJECT_ROOT / "openpocus_image_index_clean.csv"


# ----------------------------------------------------------------------
# ID NORMALIZATION
# ----------------------------------------------------------------------

def normalize_ai_id(value):
    """
    Safe identity normalization.

    IMPORTANT:
    This is equality-based only.
    It does NOT perform substring matching.

    Examples:
        10       -> numeric 10
        100      -> numeric 100
        9.19     -> numeric 9.19
        146 B    -> alpha_numeric 146b
        ED1      -> alpha_numeric ed1
    """

    s = str(value).strip().lower()

    if s.startswith("pt"):
        s = s[2:]

    s = re.sub(r"\s+", "", s)

    try:
        return ("numeric", Decimal(s).normalize())
    except InvalidOperation:
        return ("alpha_numeric", s.replace("_", ""))


def normalize_folder_name(name):
    return normalize_ai_id(name)


# ----------------------------------------------------------------------
# ZONE / FRAME EXTRACTION
# ----------------------------------------------------------------------

def extract_zone_frame(filename):
    """
    Extract zone and frame from OpenPOCUS filenames.

    Supports patterns such as:

        image_001_Pt100_z01_frame_000000.jpg
        pt200_z04_frame_000080.jpg
    """

    name = Path(filename).name

    zone_match = re.search(
        r"_z(\d+)_",
        name,
        flags=re.IGNORECASE
    )

    frame_match = re.search(
        r"_frame_(\d+)",
        name,
        flags=re.IGNORECASE
    )

    zone = int(zone_match.group(1)) if zone_match else None
    frame = int(frame_match.group(1)) if frame_match else None

    return zone, frame


# ----------------------------------------------------------------------
# IMAGE ENUMERATION
# ----------------------------------------------------------------------

def enumerate_folder(folder):

    rows = []

    # --------------------------------------------------------------
    # ZIP files
    # --------------------------------------------------------------

    zip_files = sorted(folder.glob("*.zip"))

    for zip_path in zip_files:

        with zipfile.ZipFile(zip_path, "r") as z:

            for member in z.infolist():

                if member.is_dir():
                    continue

                name = member.filename

                if not name.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                ):
                    continue

                zone, frame = extract_zone_frame(name)

                rows.append(
                    {
                        "storage_type": "ZIP_IMAGES",
                        "source_file": str(zip_path),
                        "archive_member": name,
                        "image_name": Path(name).name,
                        "zone": zone,
                        "frame": frame,
                    }
                )

    # --------------------------------------------------------------
    # Direct image files
    # --------------------------------------------------------------

    for image_path in sorted(folder.rglob("*")):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        ):
            continue

        zone, frame = extract_zone_frame(image_path.name)

        rows.append(
            {
                "storage_type": "DIRECT_IMAGES",
                "source_file": str(image_path),
                "archive_member": None,
                "image_name": image_path.name,
                "zone": zone,
                "frame": frame,
            }
        )

    return rows


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():

    print("=" * 80)
    print("BUILDING CLEAN OPENPOCUS IMAGE INDEX")
    print("=" * 80)

    splits = pd.read_csv(
        RECORD_SPLITS,
        dtype=str
    )

    folders = sorted(
        [
            p for p in DATA_ROOT.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name.lower()
    )

    print(f"\nAI IDs:             {len(splits)}")
    print(f"Raw folders:        {len(folders)}")

    if len(splits) != 226:
        raise ValueError(
            f"Expected 226 AI IDs, found {len(splits)}"
        )

    if len(folders) != 226:
        raise ValueError(
            f"Expected 226 folders, found {len(folders)}"
        )

    # ------------------------------------------------------------------
    # Build exact folder lookup.
    # ------------------------------------------------------------------

    folder_lookup = {}

    for folder in folders:

        key = normalize_folder_name(folder.name)

        if key in folder_lookup:
            raise ValueError(
                f"Duplicate folder identity: {key}"
            )

        folder_lookup[key] = folder

    # ------------------------------------------------------------------
    # Validate record table.
    # ------------------------------------------------------------------

    splits["AI ID"] = (
        splits["AI ID"]
        .astype(str)
        .str.strip()
    )

    if splits["AI ID"].duplicated().any():
        duplicates = (
            splits.loc[
                splits["AI ID"].duplicated(keep=False),
                "AI ID"
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate AI IDs in split table: {duplicates}"
        )

    # ------------------------------------------------------------------
    # Enumerate all 226 folders.
    # ------------------------------------------------------------------

    all_rows = []

    missing = []

    for _, record in splits.iterrows():

        ai_id = record["AI ID"]

        key = normalize_ai_id(ai_id)

        folder = folder_lookup.get(key)

        if folder is None:
            missing.append(ai_id)
            continue

        images = enumerate_folder(folder)

        if len(images) == 0:
            raise ValueError(
                f"No images found in folder {folder}"
            )

        print(
            f"{ai_id:>8} -> "
            f"{folder.name:<12} "
            f"{len(images):>6} images"
        )

        for image in images:

            row = {
                "record_id": ai_id,
                "folder": folder.name,
                "storage_type": image["storage_type"],
                "source_file": image["source_file"],
                "archive_member": image["archive_member"],
                "image_name": image["image_name"],
                "zone": image["zone"],
                "frame": image["frame"],
                "normal_abnormal": record[
                    "Normal or Not (0=normal; 1=abnormal)"
                ],
                "covid": record[
                    "COVID +, 0=no, 1=yes"
                ],
              "device": "",
            }

            all_rows.append(row)

    # ------------------------------------------------------------------
    # Validate mapping.
    # ------------------------------------------------------------------

    if missing:
        raise ValueError(
            "Missing folder mappings:\n"
            + "\n".join(missing)
        )

    df = pd.DataFrame(all_rows)

    unique_records = df["record_id"].nunique()
    unique_folders = df["folder"].nunique()

    print("\n" + "-" * 80)
    print("INDEX SUMMARY")
    print("-" * 80)

    print(f"Rows:            {len(df):,}")
    print(f"Unique records:  {unique_records}")
    print(f"Unique folders:  {unique_folders}")

    if unique_records != 226:
        raise ValueError(
            f"Expected 226 unique records, found {unique_records}"
        )

    if unique_folders != 226:
        raise ValueError(
            f"Expected 226 unique folders, found {unique_folders}"
        )

    # ------------------------------------------------------------------
    # Ensure each record maps to exactly one folder.
    # ------------------------------------------------------------------

    record_folder_counts = (
        df.groupby("record_id")["folder"]
        .nunique()
    )

    bad_records = record_folder_counts[
        record_folder_counts != 1
    ]

    if len(bad_records):
        raise ValueError(
            "Records mapping to multiple folders:\n"
            + bad_records.to_string()
        )

    # ------------------------------------------------------------------
    # Ensure each folder belongs to exactly one record.
    # ------------------------------------------------------------------

    folder_record_counts = (
        df.groupby("folder")["record_id"]
        .nunique()
    )

    bad_folders = folder_record_counts[
        folder_record_counts != 1
    ]

    if len(bad_folders):
        raise ValueError(
            "Folders mapping to multiple records:\n"
            + bad_folders.to_string()
        )

    # ------------------------------------------------------------------
    # Ensure no image is duplicated within the same physical source.
    # ------------------------------------------------------------------

    def image_key(row):

        if row["storage_type"] == "ZIP_IMAGES":

            return (
                f"{row['source_file']}"
                f"||"
                f"{row['archive_member']}"
            )

        return row["source_file"]

    df["image_key"] = df.apply(
        image_key,
        axis=1
    )

    duplicate_images = df[
        df.duplicated(
            "image_key",
            keep=False
        )
    ]

    print(
        f"\nDuplicate physical image entries: "
        f"{len(duplicate_images):,}"
    )

    if len(duplicate_images):

        print(
            duplicate_images[
                [
                    "record_id",
                    "folder",
                    "storage_type",
                    "source_file",
                    "archive_member",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Duplicate physical image entries detected."
        )

    # ------------------------------------------------------------------
    # Remove internal audit key before saving.
    # ------------------------------------------------------------------

    df = df.drop(columns=["image_key"])

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\n" + "-" * 80)
    print("CLEAN IMAGE INDEX CREATED")
    print("-" * 80)

    print(OUTPUT)
    print(f"Rows: {len(df):,}")
    print(f"Records: {df['record_id'].nunique()}")
    print(f"Folders: {df['folder'].nunique()}")

    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    main()
