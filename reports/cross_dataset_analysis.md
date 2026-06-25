# Cross-Dataset Generalization Analysis

## Motivation

The primary experiments in this project were conducted using train-test splits from the WELFake dataset. While these experiments provide insight into model performance on data drawn from the same distribution, they do not evaluate how well the models generalize to news articles collected from different sources.

To investigate this, the best-performing models were trained on the full WELFake dataset and evaluated on two external datasets:

* IFND
* WWFND

No retraining or fine-tuning was performed.

---

## Results

### IFND Dataset

| Model | Configuration | Accuracy |
|---------|---------:|
| Logistic Regression | Baseline | 66.66% |
| Logistic Regression | Text Only | 66.66% |
| Logistic Regression | No Source Words | 66.65% |
| Linear SVM | Baseline | 66.44% |
| Linear SVM | Hyper-Tuned | 66.54% |
| Linear SVM | Text Only | 66.36% |
| Linear SVM | Title Only | 66.65% |
| Linear SVM | No Source Words | 65.85% |

Performance on IFND was substantially lower than the 97–98% accuracy achieved on WELFake.

Most configurations predicted the majority of articles as a single class, producing extremely high recall but poor class discrimination.

---

### WWFND Dataset

| Model | Configuration | Accuracy |
|---------|---------:|
| Linear SVM | Baseline | 50.11% |
| Linear SVM | Hyper-Tuned | 50.01% |
| Linear SVM | Text Only | 50.00% |
| Linear SVM | Title Only | 49.32% |
| Linear SVM | No Source Words | 50.08% |
| Logistic Regression | Baseline | 49.35% |
| Logistic Regression | Text Only | 49.35% |
| Logistic Regression | No Source Words | 49.36% |

Performance on WWFND was close to random guessing.

Neither hyperparameter tuning nor feature-selection strategies meaningfully improved results.

---

## Analysis

Several factors likely contributed to the dramatic performance drop:

### Distribution Shift

The WELFake dataset contains long-form news articles, whereas IFND and WWFND contain much shorter news statements.

Average document lengths:

| Dataset | Average Words |
| ------- | ------------: |
| WELFake |          ~541 |
| IFND    |           ~13 |
| WWFND   |           ~18 |

The vocabulary and writing style therefore differ substantially from the data used during training.

---

### Vocabulary Mismatch

TF-IDF relies heavily on vocabulary overlap between training and testing data.

Many features learned from WELFake were absent from the external datasets, reducing the usefulness of the learned decision boundary.

---

### Source-Specific Signals

Feature importance analysis showed that publisher-related terms such as:

* Reuters
* Washington Reuters
* Breitbart

were among the strongest indicators learned by the models.

Removing source identifiers slightly reduced in-domain performance but did not substantially improve cross-dataset results.

This suggests that although publisher information contributes to performance, broader distribution differences remain the primary challenge.

---

### Model Robustness

Linear SVM remained slightly more stable than Logistic Regression under distribution shift, but the difference was small.

The dominant factor affecting performance was the feature representation rather than the classifier itself.

---

## Key Findings

* Models achieving nearly 98% accuracy on WELFake generalized poorly to external datasets.
* Cross-dataset performance dropped to approximately 66% on IFND and 50% on WWFND.
* Hyperparameter tuning had minimal effect on generalization.
* Removing source identifiers did not solve the distribution shift problem.
* Vocabulary mismatch and article-length differences appear to be the primary causes of performance degradation.
* Strong in-domain performance should not be interpreted as evidence of real-world robustness.
