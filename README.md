# Fake News Detection using Classical Machine Learning

A comparative NLP project exploring multiple text vectorization techniques and linear machine learning models for fake news classification on the WELFake dataset.

## Project Overview

This project investigates how different feature engineering strategies and traditional machine learning algorithms perform on large-scale news classification.

The focus is not only on maximizing accuracy, but also on understanding:

* The impact of vectorization methods
* The importance of source identifiers
* The contribution of article titles versus article text
* The effect of n-grams, stopword removal, and feature selection
* Training efficiency and computational trade-offs

---

## Dataset

**Dataset:** WELFake Dataset

* **Total articles:** 72,134
* **Real news:** 35,028
* **Fake news:** 37,106
* **Class distribution:** ~51.5% fake, ~48.5% real

### Features

| Column  | Description                              |
| ------- | ---------------------------------------- |
| `title` | Article headline                         |
| `text`  | Article content                          |
| `label` | Target variable (`0 = Real`, `1 = Fake`) |

### Data Quality

* Missing titles: 558
* Missing article texts: 39

The dataset is nearly balanced, making accuracy, precision, recall, and F1-score reliable evaluation metrics.

---

## Project Structure

```text
fake_news_detector/
│
├── data/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_naive_bayes.ipynb
│   └── 04_linear_svm.ipynb
│
├── reports/
│   ├── logistic_regression_analysis.md
│   ├── naive_bayes_analysis.md
│   └── linear_svm_analysis.md
│
├── results/
│   ├── logistic_regression.csv
│   ├── naive_bayes.csv
│   └── linear_svm.csv
│
├── src/
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── evaluation.py
│   └── experiments.py
│
├── requirements.txt
└── README.md
```

---

## Preprocessing Pipeline

* Missing value handling
* Text normalization
* Lowercasing
* Stopword removal
* Source identifier removal (experimental)
* Separate processing for titles and article text

---

## Feature Engineering

### Vectorization Methods

* TF-IDF Vectorization
* Count Vectorization

### Feature Configurations

* Unigrams
* Unigrams + Bigrams
* Title only
* Text only
* Combined title + text
* Separate title and text feature spaces

### Feature Selection

* `min_df` experiments (`2`, `5`, `10`, `20`)
* Maximum feature limits:

  * Title: 50,000
  * Text: 500,000

---

## Models Implemented

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (LinearSVC)

---

## Best Results

| Model               | Vectorizer       | Accuracy   | F1 Score   |
| ------------------- | ---------------- | ---------- | ---------- |
| Linear SVM          | TF-IDF           | **97.95%** | **97.95%** |
| Logistic Regression | TF-IDF           | 96.78%     | 96.78%     |
| Naive Bayes         | Count Vectorizer | 93.95%     | 93.95%     |

### Best Overall Configuration

* Model: Linear SVM
* Vectorizer: TF-IDF
* N-grams: Unigrams + Bigrams
* `min_df = 2`
* Loss: Hinge
* Separate title and text feature spaces

Performance:

* Accuracy: 97.95%
* Precision: 97.60%
* Recall: 98.40%
* F1 Score: 97.95%

---

## Cross-Dataset Evaluation

To evaluate real-world robustness, the best-performing models trained on WELFake were tested on two external fake-news datasets (IFND and WWFND).

| Dataset                     | Best Accuracy |
| --------------------------- | ------------: |
| WELFake (held-out test set) |        97.95% |
| IFND                        |        66.66% |
| WWFND                       |        50.11% |

Although the models achieved excellent performance on WELFake, accuracy dropped substantially on external datasets.

This result highlights a common challenge in fake news detection: models often learn dataset-specific vocabulary and stylistic patterns that do not transfer well to unseen sources.

Cross-dataset evaluation revealed that feature representation and dataset characteristics have a larger impact on generalization than classifier choice or hyperparameter tuning.

---

## Key Findings

* Linear models perform exceptionally well on this dataset.
* Linear SVM achieved the strongest overall performance.
* Logistic Regression provided competitive results with faster training.
* Naive Bayes benefited significantly from Count Vectorization.
* TF-IDF consistently outperformed Count Vectorization for discriminative models.
* Source identifiers contribute strongly to classification performance.
* Article text contains substantially more predictive information than titles alone.
* Separating title and text features performs better than merging them.
* Bigrams improve performance modestly despite increasing feature dimensionality.
* Increasing `min_df` reduces feature count with minimal performance loss.
* Feature extraction is substantially more expensive than model training.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fake_news_detector.git

cd fake_news_detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the WELFake dataset.

Due to file size restrictions, the dataset is not included in this repository.

1. Download the dataset from the original source.
2. Rename the file to `Dataset.csv`.
3. Place it inside the `data/` directory.

```text
data/
└── Dataset.csv
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open any notebook inside the `notebooks/` directory.

Example workflow:

1. `01_data_exploration.ipynb`
2. `02_logistic_regression.ipynb`
3. `03_naive_bayes.ipynb`
4. `04_linear_svm.ipynb`

---

## Future Work

* Evaluate on additional external fake-news datasets
* Perform domain adaptation experiments
* Investigate feature normalization techniques for improved cross-dataset robustness
* Explore word embeddings and sentence embeddings
* Compare classical ML approaches with transformer models (BERT, RoBERTa)
* Perform systematic cross-validation across multiple datasets
* Deploy the best-performing model as a web application

---

## License

This project is intended for educational and research purposes.
