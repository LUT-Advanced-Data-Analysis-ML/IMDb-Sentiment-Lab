"""Recreate the Phase 1 labelled training and validation splits in memory.

Identical training reviews are checked for label conflicts before removal.
Only review text is read from the official test file to exclude exact overlap;
test labels remain sealed for final evaluation. No split files are saved.
"""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random

import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SEED = 42
VALIDATION_FRACTION = 0.20
LABELS = (0, 1)


@dataclass(frozen=True)
class LabelledSplit:
    """Aligned review texts and sentiment labels for one model-development split"""

    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class EvaluationSplits:
    """Training and validation data plus the counts used to verify their construction"""

    train: LabelledSplit
    validation: LabelledSplit
    summary: dict


def _parquet_file(data_dir: Path, split: str) -> Path:
    """Return the single Parquet file for the given split, or raise an error"""
    matches = sorted(data_dir.glob(f"{split}-*.parquet"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {split} Parquet file; found {len(matches)}")
    return matches[0]


def _label_counts(rows: list[tuple[str, int]]) -> dict[str, int]:
    """Return a dict of label counts for the given rows. It is a public function here that I used to test the label counts in the unit tests."""
    counts = Counter(label for _, label in rows)
    return {str(label): counts[label] for label in LABELS}


def make_splits(data_dir: Path = DATA_DIR) -> EvaluationSplits:
    """Return an exact-text-disjoint, seeded 80/20 stratified split

    Original Parquet files are only read
    The unsupervised file is not used
    Any conflicting labels for identical training text cause an error
    """
    data_dir = Path(data_dir)
    source = pq.read_table(_parquet_file(data_dir, "train"), columns=["text", "label"])
    texts = source.column("text").to_pylist()
    labels = source.column("label").to_pylist()


    # 1- Check all original training rows, including reviews later excluded due to test overlap
    # 2- Stop split construction if any conflicting labels are found
    unique: dict[str, int] = {}
    for index, (text, label) in enumerate(zip(texts, labels, strict=True)):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid training review at row {index}")
        if label not in LABELS:
            raise ValueError(f"Unexpected training label {label!r} at row {index}")
        if text in unique and unique[text] != label:
            raise ValueError(f"Conflicting labels for identical training text at row {index}")
        unique[text] = label

    # Deliberately request only the text column: test labels stay untouched
    test_texts = pq.read_table(
        _parquet_file(data_dir, "test"), columns=["text"]
    ).column("text").to_pylist()
    if any(not isinstance(text, str) or not text.strip() for text in test_texts):
        raise ValueError("The test split contains an invalid review")
    test_text_set = set(test_texts)


    # 1- Remove overlapping review texts before counting duplicate removals
    # 2- Ensure each original training row has one clear exclusion reason
    overlap_rows = sum(text in test_text_set for text in texts)
    eligible = [(text, label) for text, label in unique.items() if text not in test_text_set]
    removed_duplicates = len(texts) - overlap_rows - len(eligible)

    # Shuffle each class separately to retain its proportion in the 80/20 split
    rng = Random(SEED)
    by_label = {label: [] for label in LABELS}
    for row in eligible:
        by_label[row[1]].append(row)

    train_rows: list[tuple[str, int]] = []
    validation_rows: list[tuple[str, int]] = []
    for label in LABELS:
        rows = by_label[label]
        rng.shuffle(rows)
        validation_size = round(len(rows) * VALIDATION_FRACTION)
        validation_rows.extend(rows[:validation_size])
        train_rows.extend(rows[validation_size:])
    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)

    train_text_set = {text for text, _ in train_rows}
    validation_text_set = {text for text, _ in validation_rows}
    if train_text_set & validation_text_set:
        raise AssertionError("Training and validation share exact review text")
    if (train_text_set | validation_text_set) & test_text_set:
        raise AssertionError("A training or validation review appears in test")
    if len(train_rows) + len(validation_rows) != len(eligible):
        raise AssertionError("An eligible observation was lost during splitting")
    if len(texts) != overlap_rows + removed_duplicates + len(eligible):
        raise AssertionError("Source rows do not reconcile with removals")

    summary = {
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "raw_train_rows": len(texts),
        "raw_train_class_counts": _label_counts(list(zip(texts, labels, strict=True))),
        "train_rows_excluded_for_test_overlap": overlap_rows,
        "duplicate_train_rows_removed_after_overlap": removed_duplicates,
        "eligible_unique_rows": len(eligible),
        "eligible_unique_class_counts": _label_counts(eligible),
        "training_rows": len(train_rows),
        "training_class_counts": _label_counts(train_rows),
        "validation_rows": len(validation_rows),
        "validation_class_counts": _label_counts(validation_rows),
        "official_test_rows_unchanged": len(test_texts),
        "train_validation_text_disjoint": True,
        "train_test_text_disjoint": True,
        "validation_test_text_disjoint": True,
    }
    return EvaluationSplits(
        train=LabelledSplit(*map(list, zip(*train_rows, strict=True))),
        validation=LabelledSplit(*map(list, zip(*validation_rows, strict=True))),
        summary=summary,
    )


if __name__ == "__main__":
    print(json.dumps(make_splits().summary, indent=2))
