"""Refit the approved C=10 baseline and evaluate the official test set once.

The development pool is the union of the fixed training and validation splits
after approved exact-overlap removal and deduplication. TF-IDF and logistic
regression are fitted only on that pool; test labels are used for final metrics.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)

if __package__:
    from .split_data import DATA_DIR, make_splits
    from .text_features import clean_review, make_vectorizer
else:
    from split_data import DATA_DIR, make_splits
    from text_features import clean_review, make_vectorizer


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
FINAL_C = 10.0
SEED = 42


def _official_test_rows() -> tuple[list[str], np.ndarray]:
    """Read the unchanged official test rows, including internal duplicates"""
    paths = sorted(DATA_DIR.glob("test-*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"Expected one test Parquet file; found {len(paths)}")
    table = pq.read_table(paths[0], columns=["text", "label"])
    texts = table.column("text").to_pylist()
    labels = np.asarray(table.column("label").to_pylist())
    if len(texts) != 25_000 or np.bincount(labels, minlength=2).tolist() != [12_500, 12_500]:
        raise ValueError("Official test row count or label balance changed")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Official test contains an invalid review")
    return texts, labels


def _save_confusion_matrix(matrix: np.ndarray) -> None:
    """Show exact test counts with true labels on rows and predictions on columns"""
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks((0, 1), ("Negative", "Positive"))
    ax.set_yticks((0, 1), ("Negative", "Positive"))
    ax.set(xlabel="Predicted sentiment", ylabel="True sentiment",
           title="Final logistic regression: test confusion matrix")
    threshold = matrix.max() / 2
    for row in (0, 1):
        for col in (0, 1):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black",
                    fontsize=13, fontweight="bold")
    fig.colorbar(image, ax=ax, label="Reviews", shrink=0.82)
    fig.savefig(OUTPUT_DIR / "final_test_confusion_matrix.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_roc(labels: np.ndarray, probabilities: np.ndarray, auc: float) -> None:
    """Plot discrimination across thresholds without selecting a new threshold"""
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    ax.plot(false_positive_rate, true_positive_rate, color="#386CB0", linewidth=2,
            label=f"Logistic regression (AUC = {auc:.3f})")
    ax.plot((0, 1), (0, 1), linestyle="--", color="0.5", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Final logistic regression: test ROC curve", xlim=(0, 1), ylim=(0, 1))
    ax.grid(color="0.9", linewidth=0.7)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT_DIR / "final_test_roc_curve.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_error_examples(texts: list[str], labels: np.ndarray,
                         predictions: np.ndarray, probabilities: np.ndarray) -> int:
    """Save illustrative median-confidence and high-confidence errors from each class"""
    rows = []
    for true_label in (0, 1):
        errors = np.flatnonzero((labels == true_label) & (predictions != labels))
        if not len(errors):
            continue
        confidence = np.where(predictions[errors] == 1,
                              probabilities[errors], 1 - probabilities[errors])
        ordered = errors[np.argsort(confidence)]
        for selection, index in (("median wrong-class confidence", ordered[len(ordered) // 2]),
                                 ("highest wrong-class confidence", ordered[-1])):
            rows.append({
                "test_row_index_zero_based": int(index),
                "error_type": "false positive" if true_label == 0 else "false negative",
                "selection": selection,
                "true_label": int(labels[index]),
                "predicted_label": int(predictions[index]),
                "positive_probability": float(probabilities[index]),
                "cleaned_review_excerpt": clean_review(texts[index])[:300],
            })
    with (OUTPUT_DIR / "final_test_error_examples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    """Fit the locked model, evaluate once, and save test evidence"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = make_splits()
    summary = splits.summary
    if (summary["raw_train_rows"], summary["train_rows_excluded_for_test_overlap"],
        summary["duplicate_train_rows_removed_after_overlap"],
        summary["eligible_unique_rows"]) != (25_000, 123, 96, 24_781):
        raise ValueError("Approved development-data counts changed")

    # Check that the development pool is exactly the approved 24,781 rows with the correct label counts
    development_texts = splits.train.texts + splits.validation.texts
    development_labels = np.asarray(splits.train.labels + splits.validation.labels)
    test_texts, test_labels = _official_test_rows()
    if len(set(development_texts)) != 24_781 or set(development_texts) & set(test_texts):
        raise AssertionError("Development reviews are duplicated or overlap test")
    if np.bincount(development_labels, minlength=2).tolist() != [12_337, 12_444]:
        raise AssertionError("Development labels are misaligned or incomplete")

    # Check that the cleaned development and test reviews are non-empty and disjoint
    cleaned_development = [clean_review(text) for text in development_texts]
    cleaned_test = [clean_review(text) for text in test_texts]
    if any(not text for text in cleaned_development + cleaned_test):
        raise ValueError("A review became empty during cleaning")
    cleaned_overlap = len(set(cleaned_development) & set(cleaned_test))
    if cleaned_overlap:
        raise ValueError(f"Found {cleaned_overlap} cleaned development/test overlaps")

    # Fitting ends here: test reviews only receive the fixed transformation
    vectorizer = make_vectorizer()
    development_x = vectorizer.fit_transform(cleaned_development)
    test_x = vectorizer.transform(cleaned_test)
    model = LogisticRegression(
        C=FINAL_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=SEED
    )
    model.fit(development_x, development_labels)
    if model.n_iter_[0] >= model.max_iter or list(model.classes_) != [0, 1]:
        raise RuntimeError("Final model failed to converge or has unexpected class order")
    if development_x.shape[0] != len(development_labels) or test_x.shape[0] != len(test_labels):
        raise AssertionError("Feature rows and labels are misaligned")
    if development_x.shape[1] != test_x.shape[1]:
        raise AssertionError("Final development and test feature dimensions differ")

    # Evaluate the official test set once and save all evidence
    predictions = model.predict(test_x)
    probabilities = model.predict_proba(test_x)[:, 1]
    matrix = confusion_matrix(test_labels, predictions, labels=[0, 1])
    auc = roc_auc_score(test_labels, probabilities)
    metrics = {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(test_labels, predictions, pos_label=1),
        "recall": recall_score(test_labels, predictions, pos_label=1),
        "f1": f1_score(test_labels, predictions, pos_label=1),
        "roc_auc": auc,
        "log_loss": log_loss(test_labels, model.predict_proba(test_x), labels=[0, 1]),
    }
    _save_confusion_matrix(matrix)
    _save_roc(test_labels, probabilities, auc)
    example_count = _save_error_examples(test_texts, test_labels, predictions, probabilities)

    # Save a single JSON with all final test evidence and metrics for the report
    results = {
        "model": "TF-IDF unigrams + L2 logistic regression",
        "selection": "C=10 approved from prior validation F1 and log loss; no test tuning",
        "C": FINAL_C,
        "solver": "liblinear",
        "random_state": SEED,
        "decision_threshold": 0.5,
        "development_rows": len(development_labels),
        "development_class_counts": summary["eligible_unique_class_counts"],
        "excluded_train_test_overlap_rows": summary["train_rows_excluded_for_test_overlap"],
        "duplicate_train_rows_removed_after_overlap": summary["duplicate_train_rows_removed_after_overlap"],
        "test_rows_unchanged": len(test_labels),
        "test_class_counts": {"0": int((test_labels == 0).sum()),
                              "1": int((test_labels == 1).sum())},
        "test_distinct_raw_review_texts": len(set(test_texts)),
        "cleaned_development_test_overlap": cleaned_overlap,
        "final_vocabulary_size": len(vectorizer.vocabulary_),
        "development_feature_shape": development_x.shape,
        "test_feature_shape": test_x.shape,
        "model_iterations": int(model.n_iter_[0]),
        "test_metrics": metrics,
        "confusion_matrix_labels": ["negative", "positive"],
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "illustrative_error_examples_saved": example_count,
    }
    (OUTPUT_DIR / "final_test_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
