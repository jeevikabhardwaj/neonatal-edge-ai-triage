from pathlib import Path
from io import BytesIO
import zipfile
import threading

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OpenPOCUSMinorDataset(Dataset):
    """
    OpenPOCUS dataset for the Minor project.

    Target:
        0 = Normal
        1 = Abnormal

    Images may be:
        - stored directly on disk
        - stored inside ZIP archives

    The manifest determines which images belong to
    train / validation / test.
    """

    def __init__(
        self,
        manifest_path,
        split,
        image_size=224,
        train=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.image_size = image_size
        self.train = train

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be 'train', 'validation', or 'test'"
            )

        # --------------------------------------
        # Load manifest
        # --------------------------------------
        df = pd.read_csv(
            self.manifest_path,
            low_memory=False,
        )

        df["split"] = df["split"].astype(str).str.strip()

        self.df = df[
            df["split"] == split
        ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split: {split}"
            )

        # --------------------------------------
        # Locate dataset root
        # --------------------------------------
        self.dataset_root = (
            Path.home() / "Downloads/Lung Database"
        )

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"OpenPOCUS dataset directory not found:\n"
                f"{self.dataset_root}"
            )

        # --------------------------------------
        # Image preprocessing
        # --------------------------------------
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),
                transforms.RandomRotation(
                    degrees=7
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225
                    ],
                ),
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225
                    ],
                ),
            ])

        # --------------------------------------
        # ZIP file cache
        # --------------------------------------
        self._zip_cache = {}
        self._zip_lock = threading.Lock()

    def __len__(self):
        return len(self.df)

    def _get_zip(self, zip_path):
        """
        Open a ZIP archive once and reuse it.
        """

        zip_path = str(zip_path)

        with self._zip_lock:

            if zip_path not in self._zip_cache:

                self._zip_cache[zip_path] = zipfile.ZipFile(
                    zip_path,
                    "r"
                )

            return self._zip_cache[zip_path]

    def _load_image(self, row):
        """
        Load an image either directly from disk
        or from inside a ZIP archive.
        """

        storage_type = row["storage_type"]

        # ======================================
        # DIRECT IMAGE
        # ======================================
        if storage_type == "DIRECT_IMAGES":

            image_path = Path(
                row["source_file"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found:\n{image_path}"
                )

            image = Image.open(
                image_path
            ).convert("RGB")

            return image

        # ======================================
        # ZIP IMAGE
        # ======================================
        elif storage_type == "ZIP_IMAGES":

            zip_path = Path(
                row["source_file"]
            )

            if not zip_path.exists():
                raise FileNotFoundError(
                    f"ZIP archive not found:\n{zip_path}"
                )

            member = row["archive_member"]

            archive = self._get_zip(
                zip_path
            )

            try:
                image_bytes = archive.read(
                    member
                )
            except KeyError:
                raise FileNotFoundError(
                    f"Image '{member}' not found "
                    f"inside:\n{zip_path}"
                )

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            return image

        else:
            raise ValueError(
                f"Unknown storage type: {storage_type}"
            )

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # Load image
        image = self._load_image(row)

        # Apply preprocessing
        image = self.transform(image)

        # Binary target
        label = int(row["label"])

        return {
            "image": image,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
            "record_id": str(
                row["record_id"]
            ),
            "zone": int(
                row["zone"]
            ),
            "frame": int(
                row["frame"]
            ),
        }

    def close(self):
        """
        Close all cached ZIP archives.
        """

        with self._zip_lock:

            for archive in self._zip_cache.values():
                try:
                    archive.close()
                except Exception:
                    pass

            self._zip_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_datasets(
    manifest_path=None,
    image_size=224,
):
    """
    Convenience function for creating all three datasets.
    """

    if manifest_path is None:
        manifest_path = (
            Path.cwd()
            / "data"
            / "splits"
            / "minor_frame_manifest.csv"
        )

    train_dataset = OpenPOCUSMinorDataset(
        manifest_path=manifest_path,
        split="train",
        image_size=image_size,
        train=True,
    )

    validation_dataset = OpenPOCUSMinorDataset(
        manifest_path=manifest_path,
        split="validation",
        image_size=image_size,
        train=False,
    )

    test_dataset = OpenPOCUSMinorDataset(
        manifest_path=manifest_path,
        split="test",
        image_size=image_size,
        train=False,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


if __name__ == "__main__":

    print("\n==========================================")
    print("       OPENPOCUS MINOR DATASET")
    print("==========================================")

    manifest = (
        Path.cwd()
        / "data"
        / "splits"
        / "minor_frame_manifest.csv"
    )

    train_ds, val_ds, test_ds = create_datasets(
        manifest
    )

    print(
        f"\nTrain samples:      {len(train_ds):,}"
    )

    print(
        f"Validation samples: {len(val_ds):,}"
    )

    print(
        f"Test samples:       {len(test_ds):,}"
    )

    # --------------------------------------
    # Test several samples
    # --------------------------------------
    print("\n===== SAMPLE TEST =====")

    for name, dataset, index in [
        ("TRAIN", train_ds, 0),
        ("VALIDATION", val_ds, 0),
        ("TEST", test_ds, 0),
    ]:

        sample = dataset[index]

        print(f"\n{name}")

        print(
            "Image tensor shape:",
            tuple(sample["image"].shape)
        )

        print(
            "Label:",
            sample["label"].item()
        )

        print(
            "Record:",
            sample["record_id"]
        )

        print(
            "Zone:",
            sample["zone"]
        )

        print(
            "Frame:",
            sample["frame"]
        )

        print(
            "Zone label:",
            sample["zone_label"]
        )

    # --------------------------------------
    # Test both storage types
    # --------------------------------------
    print("\n===== STORAGE TEST =====")

    for storage_type in [
        "ZIP_IMAGES",
        "DIRECT_IMAGES",
    ]:

        matches = train_ds.df[
            train_ds.df["storage_type"]
            == storage_type
        ]

        if len(matches) == 0:
            print(
                f"{storage_type}: NOT FOUND"
            )
            continue

        row_index = matches.index[0]

        sample = train_ds[row_index]

        print(
            f"{storage_type}: ✓ "
            f"Loaded successfully"
        )

        print(
            "  Shape:",
            tuple(sample["image"].shape)
        )

        print(
            "  Record:",
            sample["record_id"]
        )

    print("\n==========================================")
    print("       DATASET TEST COMPLETE")
    print("==========================================\n")
