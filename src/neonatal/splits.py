"""
Leakage-safe train/validation/test splitting for LUS datasets.

The split is performed at patient level so that videos/frames from
the same patient cannot appear in multiple partitions.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset import validate_manifest


def split_by_patient(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split a manifest into train/validation/test at patient level.

    Parameters
    ----------
    df:
        Common LUS manifest.
    test_size:
        Fraction of patients reserved for test.
    val_size:
        Fraction of the remaining patients reserved for validation.
    random_state:
        Random seed.

    Returns
    -------
    train_df, val_df, test_df
    """

    validate_manifest(df)

    patients = df["patient_id"].drop_duplicates().tolist()

    if len(patients) < 5:
        raise ValueError(
            "At least 5 unique patients are recommended for splitting."
        )

    train_patients, test_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=random_state,
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_patients, val_patients = train_test_split(
        train_patients,
        test_size=relative_val_size,
        random_state=random_state,
    )

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    # Final leakage checks.
    train_ids = set(train_df["patient_id"])
    val_ids = set(val_df["patient_id"])
    test_ids = set(test_df["patient_id"])

    if train_ids & val_ids:
        raise RuntimeError("Patient leakage detected between train and validation.")

    if train_ids & test_ids:
        raise RuntimeError("Patient leakage detected between train and test.")

    if val_ids & test_ids:
        raise RuntimeError("Patient leakage detected between validation and test.")

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir,
):
    """
    Save train/validation/test manifests.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train_manifest.csv", index=False)
    val_df.to_csv(output_dir / "validation_manifest.csv", index=False)
    test_df.to_csv(output_dir / "test_manifest.csv", index=False)


def split_summary(train_df, val_df, test_df):
    """
    Return a compact summary of the three partitions.
    """

    return {
        "train": {
            "rows": len(train_df),
            "patients": train_df["patient_id"].nunique(),
            "exams": train_df["exam_id"].nunique(),
            "videos": train_df["video_id"].nunique(),
        },
        "validation": {
            "rows": len(val_df),
            "patients": val_df["patient_id"].nunique(),
            "exams": val_df["exam_id"].nunique(),
            "videos": val_df["video_id"].nunique(),
        },
        "test": {
            "rows": len(test_df),
            "patients": test_df["patient_id"].nunique(),
            "exams": test_df["exam_id"].nunique(),
            "videos": test_df["video_id"].nunique(),
        },
    }
