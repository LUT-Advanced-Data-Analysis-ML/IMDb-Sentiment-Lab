# IMDb-Sentiment-Lab
Comparing Machine Learning Models for IMDb Review Sentiment Classification

## Phase 1 evaluation split

Install `pyarrow`, then run `python src/split_data.py` from the repository root
to print the split counts and verification results. Later code can import
`make_splits()` from `src.split_data` to obtain the labelled training and
validation reviews in memory. No derived dataset files are written.

The split procedure checks for conflicting labels on identical training text,
excludes exact train/test review overlaps from training, deduplicates the
remaining training reviews, and creates an 80/20 shuffled stratified split
with seed 42. It reads test review text only for the overlap check; test labels
are reserved for final evaluation. The unsupervised file is not used.

## Repository layout

- `data/`: original IMDb Parquet files, preserved as supplied.
- `src/`: Python source files for splitting and text features.
- `output/`: generated figures, results, and the final PDF report.

## Phase 1 text features

Install `scikit-learn` and `scipy` alongside `pyarrow`, then run
`python src/text_features.py` from the repository root. The command prints
the sparse training/validation matrix shapes and vocabulary size without
writing feature files. Later code can import `build_text_features()` from
`src.text_features` to reuse the fitted vectorizer and aligned labels.

Review cleaning decodes HTML entities, removes HTML tags, lowercases text,
and collapses whitespace. Word-level TF-IDF uses unigrams and the training
split alone for vocabulary and IDF fitting. No stop words, stemming, or
lemmatization are used. Validation reviews are only transformed; the official
test set is excluded from feature extraction at this step.

## Phase 1 exploratory figures

Install `matplotlib` alongside the dependencies above, then run
`python src/visualize_eda.py` from the repository root. The script writes
three report-ready PNG figures to `output/`: sentiment balance in the labelled
training/validation splits, training review-length distributions, and a 2D
Truncated SVD view of training TF-IDF features. The SVD is fitted on all
training reviews; a seeded sample of up to 2,000 reviews per class is plotted.

## Phase 1 logistic regression baseline

Run `python src/train_logistic_baseline.py` from the repository root. It fits
L2-regularized binary logistic regression at `C = 0.1, 1, 3, 10, 30, 100`
on training TF-IDF features. It saves training and validation accuracy,
precision, recall, F1, log loss, and generalization gaps to
`output/logistic_validation_results.csv`, plus a comparison plot in
`output/logistic_regularization_comparison.png`. The existing coefficient
table is regenerated for `C = 10` as a labelled reference, not a final model
choice. Test labels are not loaded; no complete-development refit occurs.

## Final Phase 1 baseline evaluation

Run `python src/evaluate_final_baseline.py` from the repository root after
installing the dependencies listed above. The group selected `C = 10` using
validation F1, supported by validation log loss. The script combines the
19,825 training and 4,956 validation reviews, refits the same unigram TF-IDF
configuration on all 24,781 eligible development reviews, then fits L2
logistic regression with `liblinear`, `C = 10`, and seed 42. The 123 exact
train-test overlaps and 96 repeated training rows remain excluded from the
development pool. All original Parquet files, including every one of the
25,000 official test rows, remain unchanged.

Final test metrics and counts are saved in `output/final_test_results.json`;
the confusion matrix, ROC curve, and four illustrative error excerpts are
saved beside it. Validation results in `output/logistic_validation_results.csv`
refer to the earlier 19,825/4,956 split and are distinct from final test
results. The test set is evaluated once and does not influence the model.
