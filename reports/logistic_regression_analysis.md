# Logistic Regression Analysis

## Overview

Two feature representations were evaluated using Logistic Regression:

1. TF-IDF
2. CountVectorizer

Both models used identical preprocessing and train/test splits to ensure fair comparison.

Common configuration:

- Separate title and text feature spaces
- Unigram and bigram features
- Stopword removal
- `min_df=5`
- `max_features=50,000` (title)
- `max_features=500,000` (text)

---

## Baseline Performance

### TF-IDF + Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | 96.78% |
| Precision | 95.96% |
| Recall | 97.82% |
| F1 Score | 96.78% |

### CountVectorizer + Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | 97.69% |
| Precision | 97.10% |
| Recall | 98.41% |
| F1 Score | 97.69% |

CountVectorizer consistently outperformed TF-IDF despite using identical downstream classifiers.

This suggests that raw term frequencies preserve useful information that TF-IDF downweights.

---

## Feature Importance Analysis

Inspection of Logistic Regression coefficients revealed several highly influential features.

### Real News Indicators

- `reuters`
- `breitbart`
- `said`
- `washington reuters`
- `new york`
- `york times`
- `president donald`
- `twitter`

### Fake News Indicators

- `video`
- `via`
- `breaking`
- `image`
- `image via`
- `hillary`
- `2016`
- `october`
- `trump`
- `november`

Many influential features corresponded to publisher names and source identifiers, suggesting that article provenance contributes substantially to classification performance.

Highly weighted fake-news indicators often reflected clickbait language, media-sharing terminology, and event-specific vocabulary.

---

## Impact of Source Identifiers

To evaluate whether models relied excessively on publisher information, prominent source names were removed during preprocessing.

Examples included:

- Reuters
- Breitbart
- CNN
- BBC
- Fox News
- Associated Press
- New York Times

### TF-IDF Results

| Configuration | Accuracy |
|---------------|----------|
| Baseline | 96.78% |
| Source Removal | 95.77% |

### CountVectorizer Results

| Configuration | Accuracy |
|---------------|----------|
| Baseline | 97.69% |
| Source Removal | 96.71% |

Performance decreased noticeably but remained strong.

These results suggest that source information provides meaningful signals, but neither model depends exclusively on publisher names.

Both classifiers continue to learn useful linguistic and stylistic patterns from article content.

---

## Representation Comparison

Using the same Logistic Regression classifier allowed the effect of feature representation to be isolated.

| Representation | Accuracy | F1 Score |
|----------------|----------|----------|
| TF-IDF | 96.78% | 96.78% |
| CountVectorizer | 97.69% | 97.69% |

The performance difference indicates that the primary improvement comes from the feature representation rather than the classifier itself.

This finding motivates future experiments combining alternative classifiers with both vectorization methods.

---

## N-Gram Analysis

| Configuration | Accuracy |
|---------------|----------|
| Unigrams Only | 96.72% |
| Unigrams + Bigrams | 96.78% |

Bigrams provided a modest improvement.

Although informative bigrams such as `washington reuters`, `image via`, and `new york` appeared among the top features, most predictive power originated from unigrams.

The performance gain must therefore be weighed against the substantial increase in feature dimensionality.

---

## Stopword Removal

Removing stopwords produced virtually identical results.

Because TF-IDF naturally downweights common terms, explicit stopword removal contributed little additional benefit.

For CountVectorizer, stopword removal slightly reduced feature dimensionality without affecting performance.

---

## Title vs Article Content

| Configuration | Accuracy |
|---------------|----------|
| Title Only | ~89.7% |
| Text Only | ~95.5% |
| Title + Text Combined | ~95.8% |
| Separate Title + Text Features | ~96.8% |

Article content carried most of the predictive information.

Titles alone were informative but insufficient for optimal performance.

Maintaining separate feature spaces consistently outperformed combining title and text before vectorization, suggesting that titles and article bodies contain complementary information.

---

## Effect of Minimum Document Frequency

### TF-IDF

| min_df | Features | Accuracy |
|--------|----------|----------|
| 2 | 550k | 96.75% |
| 5 | 477k | 96.78% |
| 10 | 196k | 96.85% |

### CountVectorizer

| min_df | Features | Accuracy |
|--------|----------|----------|
| 2 | 550k | 97.66% |
| 5 | 477k | 97.69% |
| 10 | 196k | 97.59% |
| 20 | 87k | 97.53% |

Increasing `min_df` substantially reduced vocabulary size with minimal performance loss.

This suggests that many rare words contribute noise rather than useful predictive information.

---

## Computational Performance

Typical execution times:

### TF-IDF

- Feature generation: 40–45 seconds
- Model training: 2–3 seconds

### CountVectorizer

- Feature generation: 40–45 seconds
- Model training: 3–5 seconds

Feature extraction consistently dominated total runtime.

Reducing feature dimensionality is therefore likely to produce larger efficiency gains than changing classifiers.

---

## Conclusions

- Logistic Regression performs exceptionally well on this dataset.
- CountVectorizer outperforms TF-IDF despite identical classifiers.
- Source identifiers contribute useful information but are not solely responsible for model performance.
- Article text is substantially more informative than titles.
- Preserving separate title and text feature spaces improves results.
- Stopword removal has minimal impact.
- Bigrams provide modest gains at significant computational cost.
- Increasing `min_df` improves efficiency with little loss in performance.
- Vectorization remains the primary computational bottleneck.

These findings establish strong baselines for future experiments involving Naive Bayes, SVMs, and embedding-based approaches.