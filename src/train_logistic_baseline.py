"""Compare L2 logistic-regression strengths on the fixed validation split.

The approved split and training-fitted TF-IDF features are reused unchanged.
Only training rows fit the classifiers. The official test labels are never
read, and no final C choice or full-development refit is made here.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

if __package__:
    from .split_data import make_splits
    from .text_features import build_text_features
else:
    from split_data import make_splits
    from text_features import build_text_features


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
C_VALUES = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
SEED = 42
TOP_WORDS = 15
COEFFICIENT_REFERENCE_C = 10.0


def _scores(labels: np.ndarray, predictions: np.ndarray,
            probabilities: np.ndarray) -> dict[str, float]:
    """Evaluate positive-class decisions and probability quality"""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
    }


def _write_results(rows: list[dict]) -> None:
    """Save each candidate's training and validation metrics for the report"""
    columns = ["C", "train_accuracy", "train_precision", "train_recall",
               "train_f1", "train_log_loss", "validation_accuracy", "validation_precision",
               "validation_recall", "validation_f1", "validation_log_loss",
               "f1_gap", "log_loss_gap", "iterations"]
    with (OUTPUT_DIR / "logistic_validation_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_top_coefficients(model: LogisticRegression, feature_names: np.ndarray) -> None:
    """Keep the previous C=10 coefficient view as an explicitly labelled reference"""
    if list(model.classes_) != [0, 1]:
        raise AssertionError("Unexpected class order for coefficient interpretation")
    coefficients = model.coef_[0]
    negative = np.argsort(coefficients)[:TOP_WORDS]
    positive = np.argsort(coefficients)[-TOP_WORDS:][::-1]
    rows = ([{"C": COEFFICIENT_REFERENCE_C, "sentiment": "negative", "word": feature_names[index],
              "coefficient": float(coefficients[index])} for index in negative]
            + [{"C": COEFFICIENT_REFERENCE_C, "sentiment": "positive", "word": feature_names[index],
                "coefficient": float(coefficients[index])} for index in positive])
    with (OUTPUT_DIR / "logistic_top_coefficients.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["C", "sentiment", "word", "coefficient"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(rows: list[dict]) -> None:
    """Show how fit and validation quality change as regularization weakens
    This function is used to check overfitting and to support the final C choice in the report."""
    values = [row["C"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric, title, ylabel in (
        (axes[0], "f1", "Positive-class F1", "F1 score"),
        (axes[1], "log_loss", "Log loss", "Log loss (lower is better)"),
    ):
        axis.plot(values, [row[f"train_{metric}"] for row in rows], "o--",
                  color="#386CB0", label="Training")
        axis.plot(values, [row[f"validation_{metric}"] for row in rows], "o-",
                  color="#E69F00", label="Validation")
        axis.set_xscale("log")
        axis.set_xticks(values, [f"{value:g}" for value in values])
        axis.set(xlabel="C (larger means weaker regularization)", ylabel=ylabel, title=title)
        axis.grid(color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "logistic_regularization_comparison.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Fit each candidate and save comparative validation evidence"""
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
        train = _scores(train_y, model.predict(features.train_x),
                        model.predict_proba(features.train_x))
        validation = _scores(validation_y, model.predict(features.validation_x),
                             model.predict_proba(features.validation_x))
        candidates.append((model, {
            "C": C,
            "train_accuracy": train["accuracy"],
            "train_precision": train["precision"],
            "train_recall": train["recall"],
            "train_f1": train["f1"],
            "train_log_loss": train["log_loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_precision": validation["precision"],
            "validation_recall": validation["recall"],
            "validation_f1": validation["f1"],
            "validation_log_loss": validation["log_loss"],
            "f1_gap": train["f1"] - validation["f1"],
            "log_loss_gap": validation["log_loss"] - train["log_loss"],
            "iterations": int(model.n_iter_[0]),
        }))

    rows = [row for _, row in candidates]
    _write_results(rows)
    _plot_comparison(rows)
    reference_model = next(model for model, row in candidates
                           if row["C"] == COEFFICIENT_REFERENCE_C)
    _write_top_coefficients(reference_model, features.vectorizer.get_feature_names_out())
    highest_f1 = max(rows, key=lambda row: (row["validation_f1"], -row["C"]))
    print(json.dumps({
        "highest_observed_validation_f1_C": highest_f1["C"],
        "final_C_choice_made": False,
        "solver": "liblinear",
        "penalty": "l2",
        "train_rows": len(train_y),
        "validation_rows": len(validation_y),
        "vocabulary_size": features.train_x.shape[1],
        "results": rows,
        "coefficient_reference_C": COEFFICIENT_REFERENCE_C,
    }, indent=2))


if __name__ == "__main__":
    main()
