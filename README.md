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
- `src/`: Python source files.
- `output/`: generated figures, results, and the final PDF report.
