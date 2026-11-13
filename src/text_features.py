"""Clean IMDb reviews and build training-fitted unigram TF-IDF features.

The approved split loader supplies labelled training and validation reviews.
Only training reviews determine the vocabulary and IDF weights; validation
reviews are transformed with that fitted vectorizer. Test data are not used.
"""

from dataclasses import dataclass
from html import unescape
import json
import re

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

if __package__:
    from .split_data import EvaluationSplits, make_splits
else:
    from split_data import EvaluationSplits, make_splits

# Regular expressions for cleaning review text
HTML_TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextFeatures:
    """Sparse features and aligned labels for the approved development splits"""

    train_x: csr_matrix
    train_y: list[int]
    validation_x: csr_matrix
    validation_y: list[int]
    vectorizer: TfidfVectorizer


def clean_review(text: str) -> str:
    """Decode entities, remove HTML tags, and normalize case and whitespace"""
    text = unescape(text)
    text = HTML_TAG.sub(" ", text)
    return WHITESPACE.sub(" ", text.lower()).strip()


def make_vectorizer() -> TfidfVectorizer:
    """Create the approved unigram TF-IDF configuration for any training set"""
    # Keep all observed unigrams, including negations
    # no stop-word list or linguistic reduction is applied before the linear baseline
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        lowercase=False,
        stop_words=None,
        min_df=1,
        max_df=1.0,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )


def build_text_features(splits: EvaluationSplits | None = None) -> TextFeatures:
    """Fit TF-IDF on training reviews and transform validation reviews only"""
    if splits is None:
        splits = make_splits()

    train_texts = [clean_review(text) for text in splits.train.texts]
    validation_texts = [clean_review(text) for text in splits.validation.texts]
    if any(not text for text in train_texts + validation_texts):
        raise ValueError("A review became empty after text cleaning")


    vectorizer = make_vectorizer()

    # 1- Fit the vectorizer on training reviews only
    # 2- Transform validation reviews with the fitted vectorizer
    # 3- Ensure the resulting feature matrices are aligned with the split labels
    train_x = vectorizer.fit_transform(train_texts)
    validation_x = vectorizer.transform(validation_texts)

    if train_x.shape[0] != len(splits.train) or validation_x.shape[0] != len(splits.validation):
        raise AssertionError("Feature rows are not aligned with split labels")
    if train_x.shape[1] != validation_x.shape[1]:
        raise AssertionError("Training and validation feature dimensions differ")

    return TextFeatures(
        train_x=train_x,
        train_y=splits.train.labels,
        validation_x=validation_x,
        validation_y=splits.validation.labels,
        vectorizer=vectorizer,
    )


if __name__ == "__main__":
    features = build_text_features()
    print(json.dumps({
        "train_shape": features.train_x.shape,
        "validation_shape": features.validation_x.shape,
        "vocabulary_size": len(features.vectorizer.vocabulary_),
        "train_nonzero_features": features.train_x.nnz,
        "validation_nonzero_features": features.validation_x.nnz,
    }, indent=2))
