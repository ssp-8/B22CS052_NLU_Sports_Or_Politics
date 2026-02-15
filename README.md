# News Category Classification (Sports vs Politics)

This folder contains a small end-to-end pipeline for building a binary news classifier using classic NLP features and linear models. The workflow is split across two notebooks and the accompanying report provides the motivation and findings:

- `preprocess_dataset.ipynb` prepares the dataset from raw JSON articles.
- `B22CS052_T4.ipynb` trains and evaluates multiple models on the prepared dataset.
- `B22CS052_Report_T4.pdf` summarizes dataset statistics, model behavior, and limitations.

## What this project does

- Reads raw news articles from `dataset/Sports_Raw_Articles/` and `dataset/Politics_Raw_Articles/`.
- Cleans the article text, filters short items, and builds a labeled CSV (`dataset.csv`).
- Splits data into train/validation/test sets (stratified).
- Compares three models (Naive Bayes, Logistic Regression, Linear SVM) with three feature types (Unigram, Bag of Words, TF-IDF).
- Reports accuracy, precision, recall, and F1 score, and saves confusion matrices.

## Folder structure

```
T4/
  B22CS052_T4.ipynb
  preprocess_dataset.ipynb
  dataset/
    dataset.csv
  results/
```

## Dataset preparation (preprocess_dataset.ipynb)

The preprocessing notebook:

- Extracts the label from the JSON `categories` field.
- Cleans text by removing common boilerplate lines, bylines, and empty lines.
- Skips articles with fewer than 50 words.
- Writes a two-column CSV with columns `text` and `label`.

Dataset source: https://github.com/Webhose/free-news-datasets

You can rerun it to regenerate `dataset/dataset.csv` if the raw data changes. After preprocessing, the report notes 1,434 usable articles (about 60% sports and 40% politics) with average article lengths around 3.3k to 3.5k characters.

## Model training and evaluation (B22CS052_T4.ipynb)

The training notebook:

- Loads `dataset/dataset.csv`.
- Splits into train/validation/test with an 80/10/10 ratio (stratified).
- Builds features using:
  - Unigram CountVectorizer
  - Bag of Words (1-2 grams)
  - TF-IDF (1-2 grams)
- Trains three models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
- Prints validation accuracy and test metrics for each model-feature pair.
- Saves confusion matrices to PNG files (one per model-feature pair).

## Key findings (from the report)

- Linear SVM with TF-IDF gave the strongest overall test performance (around 0.92 accuracy/F1), with balanced errors across the two classes.
- Multinomial Naive Bayes did best with unigram counts, which fits its independence assumptions.
- Logistic Regression was stable across features but slightly underperformed on the test set compared to its validation scores.

## How to run

1. Open `preprocess_dataset.ipynb` and run all cells to create `dataset/dataset.csv`.
2. Open `B22CS052_T4.ipynb` and run all cells to train models and view results.

## Notes

- Results depend on the random seed and the exact dataset content.
- The feature extractors use English stop words and cap the vocabulary at 10,000 terms.
- Confusion matrices are saved in the working directory as `Confusion_Matrix_<Model>_<Feature>.png`.
- The report highlights class imbalance and dataset size as limitations, along with dataset freshness for newer news trends.

## Dependencies

Typical Python dependencies used in these notebooks:

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

If needed, install them with `pip install pandas numpy scikit-learn seaborn matplotlib`.
