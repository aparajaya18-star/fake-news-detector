# Model Analysis and Experimental Findings

## Baseline Performance

The baseline model used Multinomial Naive Bayes with TF-IDF vectorization, separate feature spaces for article titles and article text, unigram and bigram features, stopword removal, and a minimum document frequency (`min_df`) of 5.

### Baseline Metrics

* Accuracy: 91.99%
* Precision: 92.58%
* Recall: 91.66%
* F1 Score: 91.99%

While the model achieved reasonably strong performance, it performed substantially worse than Logistic Regression across all evaluation metrics.

The primary advantage of Naive Bayes was computational efficiency. Model training required less than 0.1 seconds, making it significantly faster than Logistic Regression.

---

## Why Naive Bayes Underperforms

Multinomial Naive Bayes assumes that features are conditionally independent given the class label.

In text classification tasks, this assumption is often unrealistic because words frequently occur together and exhibit strong correlations.

For example, phrases such as:

* `washington reuters`
* `new york times`
* `image via`

contain highly dependent terms.

Naive Bayes treats these words as independent sources of evidence, which limits its ability to model complex relationships between features.

Logistic Regression, by contrast, learns feature weights jointly and is therefore better able to account for correlations among terms.

This difference likely explains the substantial performance gap observed between the two models.

---

## Feature Importance Analysis

Unlike Logistic Regression, which provides signed feature coefficients, Naive Bayes estimates class-conditional probabilities.

Feature importance was calculated using the difference between log probabilities for each class.

### Real News Indicators

* `washington reuters`
* `reuters us`
* `york times`
* `factbox`
* `rohingya`

### Fake News Indicators

* `image via`
* `century wire`
* `wow`
* `boiler room`
* `boiler`

Many influential features corresponded to publisher names, source identifiers, and recurring phrases associated with specific news outlets.

This suggests that the model relies heavily on identifying characteristic vocabulary rather than learning broader linguistic patterns.

---

## Vectorization Analysis

A comparison between TF-IDF and CountVectorizer revealed a consistent advantage for raw word counts.

| Configuration           | Accuracy |
| ----------------------- | -------- |
| TF-IDF                  | 91.99%   |
| CountVectorizer         | 92.77%   |
| CountVectorizer + α=0.1 | 93.95%   |

This behavior is expected because Multinomial Naive Bayes models token occurrence frequencies directly.

TF-IDF transforms raw counts by applying inverse document frequency weighting and normalization, which weakens the probabilistic assumptions underlying the model.

As a result, CountVectorizer produced superior performance across all experiments.

---

## Effect of Smoothing

The smoothing parameter α controls how strongly the model adjusts feature probabilities to avoid zero-frequency issues.

Several values were evaluated:

| Alpha | Accuracy |
| ----- | -------- |
| 0.1   | 93.04%   |
| 0.5   | 92.35%   |
| 1.0   | 91.99%   |
| 2.0   | 91.54%   |

Lower smoothing values consistently improved performance.

This suggests that the dataset is sufficiently large that aggressive smoothing is unnecessary.

Excessive smoothing reduces the influence of highly informative features by making class probabilities more uniform.

Among the tested values, α = 0.1 provided the best overall results.

---

## N-Gram Analysis

Removing bigrams significantly reduced performance.

| Configuration     | Accuracy |
| ----------------- | -------- |
| TF-IDF + Unigrams | 89.89%   |
| Count + Unigrams  | 90.84%   |

The performance reduction was noticeably larger than the decrease observed for Logistic Regression.

This suggests that bigrams are especially valuable for Naive Bayes because they partially compensate for the model's independence assumption.

Phrases such as:

* `new york`
* `york times`
* `image via`
* `washington reuters`

provide meaningful context that individual words cannot capture effectively.

---

## Impact of Source Identifiers

To evaluate whether the model relied excessively on publisher information, several source phrases were removed during preprocessing.

Examples included:

* Reuters
* Breitbart
* CNN
* BBC
* Fox News
* Associated Press
* New York Times

Performance decreased substantially:

* Accuracy: 91.99% → 89.34%
* F1 Score: 91.99% → 89.33%

The magnitude of this decrease was considerably larger than the corresponding drop observed for Logistic Regression.

This indicates that Naive Bayes depends more heavily on highly discriminative tokens such as publisher names and source identifiers.

In contrast, Logistic Regression appears to learn more distributed patterns across many correlated features.

---

## Computational Performance

Training time for Naive Bayes was extremely low.

Typical timings were:

* Feature Generation: 45–55 seconds
* Model Training: 0.04–0.15 seconds

As with Logistic Regression, vectorization remained the dominant computational bottleneck.

Reducing feature dimensionality had a much larger impact on overall runtime than changing the classifier itself.

---

## Key Conclusions

* Multinomial Naive Bayes performed substantially worse than Logistic Regression.
* CountVectorizer consistently outperformed TF-IDF.
* Lower smoothing values improved performance.
* Bigrams provided meaningful gains, particularly for Naive Bayes.
* The model relied heavily on source identifiers and publisher names.
* Training time was extremely low, making Naive Bayes computationally efficient.
* Feature extraction remained the primary runtime bottleneck.

Overall, Naive Bayes provides a strong and computationally inexpensive baseline, but its simplifying assumptions limit performance on this dataset compared with discriminative models such as Logistic Regression.