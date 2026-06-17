# Linear SVM Analysis

## Model Overview

Linear Support Vector Machines were evaluated using both TF-IDF and Count Vectorizer representations. Similar to previous experiments, article titles and body text were vectorized separately before being combined into a single sparse feature matrix.

The primary objective was to determine whether maximizing the decision margin through a Support Vector Machine could improve performance over Logistic Regression while maintaining computational efficiency.

---

## Baseline Performance

The baseline configuration used:

* TF-IDF vectorization
* Separate feature spaces for titles and article text
* Stopword removal
* Unigram and bigram features
* `min_df=5`
* Default regularization (`C=1.0`)

### Baseline Metrics

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 97.84% |
| Precision | 97.29% |
| Recall    | 98.51% |
| F1 Score  | 97.84% |

The baseline Linear SVM outperformed Logistic Regression across all evaluation metrics, although the improvement was relatively small.

Recall remained consistently higher than precision, indicating that the model prioritized minimizing false negatives while maintaining a low false-positive rate.

---

## Feature Importance Analysis

Inspection of the learned coefficients revealed feature rankings similar to those observed for Logistic Regression.

### Real News Indicators

* `reuters`
* `said`
* `washington reuters`
* `breitbart`
* `trumps`

### Fake News Indicators

* `via`
* `video`
* `image`
* `image via`
* `breaking`

The overlap between important features across Logistic Regression and Linear SVM suggests that both models learn similar decision boundaries within the TF-IDF feature space.

This consistency indicates that performance gains arise primarily from the feature representation rather than from differences between the classifiers themselves.

---

## Effect of Regularization Parameter (C)

Different values of the regularization parameter were evaluated to examine the trade-off between margin size and classification accuracy.

| C    | Accuracy |
| ---- | -------: |
| 0.01 |   94.77% |
| 0.1  |   97.19% |
| 0.5  |   97.83% |
| 1.0  |   97.84% |
| 2.0  |   97.86% |
| 10.0 |   97.89% |

Very small values of `C` resulted in underfitting due to excessive regularization.

Performance improved rapidly as `C` increased, although gains beyond the default value were minimal.

Higher values of `C` also increased training time considerably.

---

## Effect of Minimum Document Frequency

Different values of `min_df` were evaluated to determine the impact of removing rare terms.

| min_df | Features | Accuracy |
| ------ | -------: | -------: |
| 2      |  550,000 |   97.93% |
| 5      |  477,076 |   97.84% |
| 10     |  195,987 |   97.91% |
| 20     |   87,429 |   97.82% |

Lower values of `min_df` slightly improved performance by retaining more rare terms.

However, increasing `min_df` dramatically reduced feature dimensionality with only negligible decreases in accuracy.

The `min_df=10` configuration reduced the feature space by nearly 60% while maintaining performance comparable to the best model.

---

## N-Gram Analysis

The contribution of bigram features was evaluated by comparing unigram-only models with unigram-plus-bigram models.

| Configuration      | Accuracy |
| ------------------ | -------: |
| Unigrams Only      |   97.58% |
| Unigrams + Bigrams |   97.84% |

The inclusion of bigrams consistently improved performance.

Important phrases such as `washington reuters` and `image via` provided contextual information unavailable to unigram-only models.

---

## Impact of Source Identifiers

To determine whether the classifier relied excessively on publisher information, common source names were removed during preprocessing.

Performance decreased from:

* Accuracy: 97.84% → 97.15%
* F1 Score: 97.84% → 97.15%

Although source identifiers contribute meaningful information, the performance decrease was relatively modest.

This suggests that the model primarily learns linguistic patterns associated with fake and real news rather than simply memorizing publishers.

---

## TF-IDF vs Count Vectorizer

Linear SVM performed substantially better with TF-IDF features than with raw term counts.

| Vectorizer       | Best Accuracy |
| ---------------- | ------------: |
| TF-IDF           |        97.95% |
| Count Vectorizer |        97.61% |

TF-IDF weighting suppresses extremely frequent words while emphasizing informative terms.

In contrast, Count Vectorizer produced larger feature magnitudes and required significantly longer training times.

For example:

| Vectorizer       | Training Time |
| ---------------- | ------------: |
| TF-IDF           |        ~0.9 s |
| Count Vectorizer |         ~97 s |

These findings indicate that TF-IDF produces a feature representation that is both more informative and more computationally efficient for Linear SVM.

---

## Best Configuration

The strongest overall configuration used:

* TF-IDF vectorization
* Unigrams and bigrams
* `min_df=2`
* Hinge loss
* `C=10`

### Final Metrics

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 97.95% |
| Precision | 97.60% |
| Recall    | 98.40% |
| F1 Score  | 97.95% |

Despite achieving the highest score, the improvement over the baseline configuration was relatively small.

This suggests that the baseline model was already close to optimal.

---

## Comparison with Other Models

| Model               | Best F1 Score |
| ------------------- | ------------: |
| Naive Bayes         |        93.95% |
| Logistic Regression |        97.69% |
| Linear SVM          |        97.95% |

The small difference between Logistic Regression and Linear SVM indicates that the dataset is close to linearly separable in TF-IDF space.

Naive Bayes performed substantially worse, suggesting that the conditional independence assumption is too restrictive for this task.

---

## Key Conclusions

* Linear SVM achieved the strongest overall performance.
* TF-IDF features consistently outperformed Count Vectorizer.
* Performance improvements from hyperparameter tuning were relatively small.
* Bigrams provided meaningful contextual information.
* Source identifiers contributed useful signals but were not the primary reason for model success.
* Increasing `min_df` substantially reduced feature dimensionality with minimal performance loss.
* Feature engineering had a greater impact on performance than classifier selection.
* The strong performance of both Logistic Regression and Linear SVM indicates that the dataset is nearly linearly separable in TF-IDF space.