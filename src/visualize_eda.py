"""Create Phase 1 figures from the approved labelled development splits.

Class balance is shown for training and validation. Review lengths and the
two-dimensional TF-IDF view use training reviews only. The official test set
is accessed by the split loader for exact-overlap exclusion, not for plots.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

if __package__:
    from .split_data import make_splits
    from .text_features import build_text_features, clean_review
else:
    from split_data import make_splits
    from text_features import build_text_features, clean_review


# Constants for the EDA figures and printed summary
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
SEED = 42
SAMPLE_PER_CLASS = 2000
COLORS = {0: "#386CB0", 1: "#E69F00"}
NAMES = {0: "Negative", 1: "Positive"}


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure to the output directory. It is a public function"""
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_class_distribution(train_labels: np.ndarray, validation_labels: np.ndarray) -> None:
    """Show sentiment proportions and sample sizes in both labelled splits"""
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    for y, (name, labels) in enumerate((('Training', train_labels), ('Validation', validation_labels))):
        counts = np.bincount(labels, minlength=2)
        percentages = 100 * counts / counts.sum()
        left = 0.0
        for label in (0, 1):
            ax.barh(y, percentages[label], left=left, color=COLORS[label],
                    height=0.55, label=NAMES[label] if y == 0 else None)
            ax.text(left + percentages[label] / 2, y, f"{percentages[label]:.1f}%",
                    ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            left += percentages[label]
        ax.text(101, y, f"n = {counts.sum():,}", va="center", fontsize=9)
    ax.set_yticks((0, 1), ("Training", "Validation"))
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xlabel("Share of reviews (%)")
    ax.set_title("Sentiment balance in labelled development data")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, "eda_class_distribution.png")


def plot_review_lengths(texts: list[str], labels: np.ndarray) -> dict[str, float]:
    """Compare cleaned whitespace word counts without hiding long reviews"""
    lengths = np.fromiter((len(clean_review(text).split()) for text in texts), dtype=int)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    medians = {}
    for label in (0, 1):
        values = np.sort(lengths[labels == label])
        median = float(np.median(values))
        medians[NAMES[label].lower()] = median
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                color=COLORS[label], linewidth=2,
                label=f"{NAMES[label]} (median {median:.0f} words)")
    ax.set(xlabel="Review length (cleaned, whitespace-separated words)",
           ylabel="Cumulative share of reviews",
           title="Training review lengths by sentiment", ylim=(0, 1.01))
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, "eda_review_lengths.png")
    return medians


def plot_svd(train_x, labels: np.ndarray) -> dict[str, object]:
    """Fit two SVD components on training TF-IDF and plot a stratified sample"""
    reducer = TruncatedSVD(n_components=2, n_iter=7, random_state=SEED)
    coordinates = reducer.fit_transform(train_x)

    # Sample only for the display; the projection is fitted on all training rows
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        chosen = rng.choice(candidates, size=min(SAMPLE_PER_CLASS, len(candidates)), replace=False)
        ax.scatter(coordinates[chosen, 0], coordinates[chosen, 1], s=10,
                   alpha=0.35, color=COLORS[label], label=f"{NAMES[label]} (n={len(chosen):,})",
                   edgecolors="none", rasterized=True)
    explained = reducer.explained_variance_ratio_
    ax.set(xlabel=f"SVD component 1 ({100 * explained[0]:.1f}% variance)",
           ylabel=f"SVD component 2 ({100 * explained[1]:.1f}% variance)",
           title="Training TF-IDF reviews projected into two dimensions")
    ax.legend(loc="best", frameon=True, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, "eda_tfidf_svd.png")
    return {"svd_explained_variance_ratio": explained.tolist(),
            "plotted_reviews_per_class": SAMPLE_PER_CLASS}


def main() -> None:
    """Generate the three EDA figures and print the settings used."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = make_splits()
    features = build_text_features(splits)
    train_labels = np.asarray(splits.train.labels)
    validation_labels = np.asarray(splits.validation.labels)
    plot_class_distribution(train_labels, validation_labels)
    medians = plot_review_lengths(splits.train.texts, train_labels)
    svd_summary = plot_svd(features.train_x, train_labels)
    print(json.dumps({
        "training_rows": len(train_labels),
        "validation_rows": len(validation_labels),
        "training_class_counts": splits.summary["training_class_counts"],
        "validation_class_counts": splits.summary["validation_class_counts"],
        "median_training_review_words": medians,
        "tfidf_shape": features.train_x.shape,
        "svd_seed": SEED,
        **svd_summary,
        "figures": ["eda_class_distribution.png", "eda_review_lengths.png", "eda_tfidf_svd.png"],
    }, indent=2))


if __name__ == "__main__":
    main()
