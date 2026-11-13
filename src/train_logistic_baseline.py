"""Select an L2 logistic-regression baseline on the fixed validation split.

The approved split and training-fitted TF-IDF features are reused unchanged.
Only training rows fit the classifiers; validation F1 selects C. The official
test labels are never read and no final-data refit is performed here.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if __package__:
    from .split_data import make_splits
    from .text_features import build_text_features
else:
    from split_data import make_splits
    from text_features import build_text_features


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
C_VALUES = (0.1, 1.0, 10.0)
SEED = 42
TOP_WORDS = 15


def _scores(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Evaluate the positive class (label 1) at the default 0.5 threshold"""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
    }


def _write_results(rows: list[dict]) -> None:
    """Save each candidate's training and validation metrics for the report"""
    columns = ["C", "selected", "train_accuracy", "train_precision", "train_recall",
               "train_f1", "validation_accuracy", "validation_precision",
               "validation_recall", "validation_f1", "iterations"]
    with (OUTPUT_DIR / "logistic_validation_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_top_coefficients(model: LogisticRegression, feature_names: np.ndarray) -> list[dict]:
    """Save the strongest signed associations with positive and negative sentiment"""
    if list(model.classes_) != [0, 1]:
        raise AssertionError("Unexpected class order for coefficient interpretation")
    coefficients = model.coef_[0]
    negative = np.argsort(coefficients)[:TOP_WORDS]
    positive = np.argsort(coefficients)[-TOP_WORDS:][::-1]
    rows = ([{"sentiment": "negative", "word": feature_names[index],
              "coefficient": float(coefficients[index])} for index in negative]
            + [{"sentiment": "positive", "word": feature_names[index],
                "coefficient": float(coefficients[index])} for index in positive])
    with (OUTPUT_DIR / "logistic_top_coefficients.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["sentiment", "word", "coefficient"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    """
    Fit the candidate models.
    Select C.
    Save validation summaries.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = make_splits()
    features = build_text_features(splits)
    train_y = np.asarray(features.train_y)
    validation_y = np.asarray(features.validation_y)
    if not np.array_equal(train_y, splits.train.labels) or not np.array_equal(
        validation_y, splits.validation.labels
    ):
        raise AssertionError("Feature rows and split labels are misaligned")
    if features.train_x.shape[0] != len(train_y) or features.validation_x.shape[0] != len(validation_y):
        raise AssertionError("Feature and label row counts differ")

    candidates = []
    for C in C_VALUES:
        model = LogisticRegression(
            C=C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED
        )
        model.fit(features.train_x, train_y)
        if model.n_iter_[0] >= model.max_iter:
            raise RuntimeError(f"Logistic regression did not converge for C={C}")
        train = _scores(train_y, model.predict(features.train_x))
        validation = _scores(validation_y, model.predict(features.validation_x))
        candidates.append((model, {
            "C": C,
            "train_accuracy": train["accuracy"],
            "train_precision": train["precision"],
            "train_recall": train["recall"],
            "train_f1": train["f1"],
            "validation_accuracy": validation["accuracy"],
            "validation_precision": validation["precision"],
            "validation_recall": validation["recall"],
            "validation_f1": validation["f1"],
            "iterations": int(model.n_iter_[0]),
        }))

    # Smaller C means stronger regularization, so it breaks an exact F1 tie
    selected_model, selected_row = max(
        candidates, key=lambda candidate: (candidate[1]["validation_f1"], -candidate[1]["C"])
    )
    rows = [{**row, "selected": row is selected_row} for _, row in candidates]
    _write_results(rows)
    coefficients = _write_top_coefficients(
        selected_model, features.vectorizer.get_feature_names_out()
    )
    print(json.dumps({
        "selected_C": selected_row["C"],
        "selection_metric": "validation F1 for positive class; smaller C breaks exact ties",
        "solver": "liblinear",
        "penalty": "l2",
        "train_rows": len(train_y),
        "validation_rows": len(validation_y),
        "vocabulary_size": features.train_x.shape[1],
        "results": rows,
        "top_negative_words": [item["word"] for item in coefficients[:TOP_WORDS]],
        "top_positive_words": [item["word"] for item in coefficients[TOP_WORDS:]],
    }, indent=2))


if __name__ == "__main__":
    main()
