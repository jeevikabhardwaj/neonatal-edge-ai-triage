"""
Common dataset utilities for pediatric/neonatal LUS data.

The pipeline is designed to support multiple datasets such as:
- PedLUS
- Fatima et al. neonatal LUS

Dataset-specific preprocessing should convert raw metadata into
the common manifest format before model training.
"""

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "dataset",
    "patient_id",
    "exam_id",
    "video_id",
    "frame_path",
    "lus_score",
    "label_type",
]


def validate_manifest(df: pd.DataFrame) -> None:
    """
    Validate the common LUS manifest.

    Raises:
        ValueError: if required columns or label constraints are invalid.
    """

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Manifest is empty.")

    # LUS scores must be 0–3.
    invalid_scores = ~df["lus_score"].isin([0, 1, 2, 3])

    if invalid_scores.any():
        invalid_values = (
            df.loc[invalid_scores, "lus_score"]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Invalid LUS scores found: {invalid_values}"
        )

    # Prevent accidental frame-level leakage.
    if df["patient_id"].isna().any():
        raise ValueError("Missing patient_id values detected.")

    if df["exam_id"].isna().any():
        raise ValueError("Missing exam_id values detected.")

    if df["video_id"].isna().any():
        raise ValueError("Missing video_id values detected.")

    valid_label_types = {"frame", "video"}

    invalid_label_types = (
        set(df["label_type"].dropna().unique())
        - valid_label_types
    )

    if invalid_label_types:
        raise ValueError(
            f"Invalid label_type values: {invalid_label_types}"
        )


def load_manifest(path):
    """
    Load and validate a CSV manifest.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    df = pd.read_csv(path)

    validate_manifest(df)

    return df


def manifest_summary(df: pd.DataFrame) -> dict:
    """
    Return useful dataset statistics.
    """

    validate_manifest(df)

    return {
        "rows": len(df),
        "patients": df["patient_id"].nunique(),
        "exams": df["exam_id"].nunique(),
        "videos": df["video_id"].nunique(),
        "datasets": df["dataset"].unique().tolist(),
        "label_distribution": (
            df["lus_score"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "label_types": (
            df["label_type"]
            .value_counts()
            .to_dict()
        ),
    }
