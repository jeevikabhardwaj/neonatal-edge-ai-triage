"""
Common lung-ultrasound severity labels.

These labels describe LUS imaging severity.
They must NOT be interpreted directly as clinical triage risk.
"""

LUS_LABELS = {
    0: "Score 0",
    1: "Score 1",
    2: "Score 2",
    3: "Score 3",
}

LUS_DESCRIPTIONS = {
    0: "Normal/aerated LUS pattern",
    1: "Mild interstitial abnormality",
    2: "Moderate/severe interstitial abnormality",
    3: "Consolidation/severe LUS abnormality",
}


def validate_lus_score(score):
    """Return a validated integer LUS score."""
    score = int(score)

    if score not in LUS_LABELS:
        raise ValueError(
            f"Invalid LUS score {score}. Expected 0, 1, 2, or 3."
        )

    return score


def get_lus_label(score):
    """Return human-readable label for a LUS score."""
    score = validate_lus_score(score)
    return LUS_LABELS[score]


def get_lus_description(score):
    """Return description for a LUS score."""
    score = validate_lus_score(score)
    return LUS_DESCRIPTIONS[score]
