from pathlib import Path
from decimal import Decimal, InvalidOperation
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/Users/D/Downloads/Lung Database")
RECORD_SPLITS = PROJECT_ROOT / "openpocus_record_splits.csv"


def clean_text(value):
    return str(value).strip().lower()


def normalize_identifier(value):
    """
    Convert an AI ID / folder identifier into a safe comparable form.

    Examples:
        10       -> ("numeric", Decimal("10"))
        100      -> ("numeric", Decimal("100"))
        9.19     -> ("numeric", Decimal("9.19"))
        31.2     -> ("numeric", Decimal("31.2"))
        146 B    -> ("alpha_numeric", "146b")
        ED1      -> ("alpha_numeric", "ed1")

    Crucially:
        10 != 100
        20 != 200
        25 != 205
    """

    s = clean_text(value)

    # Remove the Pt prefix if present.
    if s.startswith("pt"):
        s = s[2:]

    # Remove whitespace.
    s = re.sub(r"\s+", "", s)

    # Numeric IDs, including decimals.
    try:
        number = Decimal(s)
        return ("numeric", number.normalize())
    except InvalidOperation:
        pass

    # Non-numeric identifiers such as 146A / 146 B.
    s = s.replace("_", "")

    return ("alpha_numeric", s)


def folder_identifier(folder_name):
    """
    Convert a raw folder name into the same safe identity representation.
    """

    s = clean_text(folder_name)

    if s.startswith("pt"):
        s = s[2:]

    s = re.sub(r"\s+", "", s)

    # ED1, ED2, ...
    if s.startswith("ed"):
        return ("alpha_numeric", s)

    # 146A, 146B, etc.
    try:
        Decimal(s)
        return ("numeric", Decimal(s).normalize())
    except InvalidOperation:
        return ("alpha_numeric", s)


def main():

    print("=" * 80)
    print("SAFE RAW FOLDER ↔ AI ID MAPPING CHECK")
    print("=" * 80)

    splits = pd.read_csv(
        RECORD_SPLITS,
        dtype=str
    )

    folders = sorted(
        [p for p in DATA_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )

    ai_ids = (
        splits["AI ID"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    print(f"\nAI IDs in split table: {len(ai_ids)}")
    print(f"Raw dataset folders:   {len(folders)}")

    if len(ai_ids) != 226:
        raise ValueError(
            f"Expected 226 AI IDs, found {len(ai_ids)}"
        )

    if len(folders) != 226:
        raise ValueError(
            f"Expected 226 folders, found {len(folders)}"
        )

    # ------------------------------------------------------------------
    # Build folder identity map.
    # ------------------------------------------------------------------

    folder_map = {}

    for folder in folders:

        key = folder_identifier(folder.name)

        if key in folder_map:
            raise ValueError(
                f"DUPLICATE FOLDER IDENTITY:\n"
                f"{folder_map[key]}\n"
                f"{folder.name}\n"
                f"Identity: {key}"
            )

        folder_map[key] = folder

    # ------------------------------------------------------------------
    # Map every AI ID.
    # ------------------------------------------------------------------

    mappings = []
    missing = []
    ambiguous = []

    for ai_id in ai_ids:

        key = normalize_identifier(ai_id)

        candidates = []

        for folder_key, folder in folder_map.items():
            if folder_key == key:
                candidates.append(folder)

        if len(candidates) == 0:
            missing.append((ai_id, key))

        elif len(candidates) > 1:
            ambiguous.append(
                (ai_id, key, [p.name for p in candidates])
            )

        else:
            mappings.append(
                {
                    "AI ID": ai_id,
                    "folder": candidates[0].name,
                    "identity": str(key),
                }
            )

    # ------------------------------------------------------------------
    # Print results.
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("MAPPING RESULTS")
    print("-" * 80)

    print(f"Successfully mapped: {len(mappings)}")
    print(f"Missing mappings:    {len(missing)}")
    print(f"Ambiguous mappings:  {len(ambiguous)}")

    if missing:
        print("\nMISSING MAPPINGS:")
        for item in missing:
            print(item)

    if ambiguous:
        print("\nAMBIGUOUS MAPPINGS:")
        for item in ambiguous:
            print(item)

    # ------------------------------------------------------------------
    # Important collision checks.
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("CRITICAL ID COLLISION CHECK")
    print("-" * 80)

    dangerous_pairs = [
        ("10", "100"),
        ("20", "200"),
        ("25", "205"),
        ("26", "206"),
        ("11", "101"),
        ("12", "102"),
        ("13", "103"),
        ("15", "105"),
        ("16", "106"),
        ("18", "108"),
        ("19", "109"),
        ("21", "201"),
        ("22", "202"),
        ("24", "204"),
    ]

    mapping_dict = {
        str(x["AI ID"]): x["folder"]
        for x in mappings
    }

    all_safe = True

    for a, b in dangerous_pairs:

        fa = mapping_dict.get(a)
        fb = mapping_dict.get(b)

        print(f"{a:>5} -> {str(fa):<15} | {b:>5} -> {str(fb):<15}")

        if fa is None or fb is None or fa == fb:
            all_safe = False

    print("\n" + "-" * 80)
    print("FINAL CHECK")
    print("-" * 80)

    if (
        len(missing) == 0
        and len(ambiguous) == 0
        and len(mappings) == 226
        and all_safe
        and len(set(x["folder"] for x in mappings)) == 226
    ):
        print("[PASS] All 226 AI IDs map to exactly one unique raw folder.")
        print("[PASS] No dangerous ID collisions detected.")
        print("[PASS] 226 unique folders are mapped.")
        print("\nSAFE TO REBUILD IMAGE INDEX.")
    else:
        print("[FAIL] Mapping is NOT safe yet.")
        print("DO NOT rebuild the image index.")

    # ------------------------------------------------------------------
    # Show all mappings.
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("ALL MAPPINGS")
    print("-" * 80)

    for x in mappings:
        print(f"{x['AI ID']:>8} -> {x['folder']}")

    print("=" * 80)


if __name__ == "__main__":
    main()
