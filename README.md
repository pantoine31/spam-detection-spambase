# Spam Email Detection with Machine Learning — Spambase Dataset

> Σύγκριση Αλγορίθμων Μηχανικής Μάθησης για την Ανίχνευση Ανεπιθύμητης Ηλεκτρονικής Αλληλογραφίας

A comparative study of classical and modern machine learning algorithms for binary spam classification on the UCI **Spambase** dataset, with special emphasis on minimizing **false positives** (legitimate emails misclassified as spam).

---

## Authors

- **Antonios Papakonstantinou** — antonhspap@icloud.com
- **Dimitrios Kostiris** — dimkostir@gmail.com

M.Sc. *"Information Systems and Services"*, University of Piraeus

---

## Overview

Spam filtering is a classic binary classification problem where misclassifying a legitimate email as spam (a *false positive*) is far more costly than the reverse, since the user may permanently lose important messages. This project trains, evaluates, and tunes six different classifiers on the Spambase dataset and identifies the configuration that best balances overall accuracy with a low false-positive rate.

The final tuned **Random Forest** model reaches **~95.11% accuracy** with only **15 false positives** on the test set.

---

## Dataset

- **Source:** UCI Machine Learning Repository — Spambase
- **Size:** 4,601 emails
- **Features:** 57 numeric attributes describing:
  - Frequency of specific keywords (e.g. *free*, *money*, *credit*)
  - Frequency of special characters (`$`, `!`, etc.)
  - Statistics on runs of capital letters
- **Target:** Binary — `1` = spam, `0` = non-spam (ham)
- **Class distribution:** 60.6% non-spam / 39.4% spam — fairly balanced
- **Missing values:** None

---

## Methodology

1. **Data exploration** — class distribution and feature inspection.
2. **Train/test split** — 80/20 with **stratified sampling** to preserve class proportions.
3. **Scaling** — `StandardScaler` applied (fit on train, transform on test) for distance- and gradient-based models (Logistic Regression, SVM, k-NN). Avoids data leakage.
4. **Model training** — six classifiers compared.
5. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrix, with **special focus on false positives**.
6. **Hyperparameter tuning** — `GridSearchCV` with cross-validation on the two best-performing models (Random Forest, Gradient Boosting).

---

## Algorithms Compared

| # | Algorithm | Role |
|---|-----------|------|
| 1 | Logistic Regression | Linear baseline |
| 2 | Naive Bayes (Gaussian) | Probabilistic, common in spam filtering |
| 3 | Support Vector Machine (RBF) | Margin-based, handles non-linear boundaries |
| 4 | k-Nearest Neighbors | Non-parametric, distance-based |
| 5 | Random Forest | Ensemble of decision trees |
| 6 | Gradient Boosting (`HistGradientBoostingClassifier`) | Sequential boosting |

---

## Results — Baseline Models

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.9511** | 0.9441 | 0.9311 | 0.9376 | 20 | 25 |
| Random Forest | 0.9457 | 0.9510 | 0.9091 | 0.9296 | **17** | 33 |
| SVM (RBF) | 0.9273 | 0.9277 | 0.8843 | 0.9055 | 25 | 42 |
| Logistic Regression | 0.9294 | 0.9209 | 0.8981 | 0.9093 | 28 | 37 |
| k-NN (k=5) | 0.9077 | 0.8861 | 0.8788 | 0.8824 | 41 | 44 |
| Naive Bayes | 0.8339 | 0.7178 | 0.9532 | 0.8189 | 136 | 17 |

**Key observations:**
- Tree-based / boosting methods clearly dominate.
- **Naive Bayes** has the highest spam recall but produces 136 false positives — unusable in practice.
- Linear classifiers offer a stable but inferior baseline.

---

## Hyperparameter Tuning (GridSearchCV)

### Random Forest — improved ✅

Best parameters found:
```python
{
  "n_estimators":      200,
  "max_depth":         None,
  "min_samples_split": 2,
  "max_features":      "log2"
}
```

Best CV score: **~0.9418**

| Metric | Default RF | Tuned RF |
|---|---:|---:|
| Accuracy | 0.9457 | **0.9511** |
| False Positives | 17 | **15** |
| F1 (spam) | 0.9296 | 0.9367 |

### Gradient Boosting — did NOT improve on test set ❌

Best parameters found:
```python
{
  "learning_rate": 0.1,
  "max_depth":     9,
  "max_iter":      200
}
```

Best CV score: **~0.9464**

| Metric | Default GB | Tuned GB |
|---|---:|---:|
| Accuracy | **0.9511** | 0.9457 |
| False Positives | **20** | 28 |

A textbook example that strong cross-validation performance does not guarantee better test-set generalization — likely mild overfitting to the CV folds.

---

## Final Model

🏆 **Tuned Random Forest** — best overall balance of accuracy and minimal false positives, which is the priority metric for spam filtering.

Final confusion matrix:

|  | Predicted: ham | Predicted: spam |
|---|---:|---:|
| **Actual: ham**  | 543 (TN) | 15 (FP) |
| **Actual: spam** | 30 (FN)  | 333 (TP) |

---

## Requirements

```txt
python >= 3.9
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install:
```bash
pip install -r requirements.txt
```

---

## Conclusions

- Tree-based / boosting methods outperform simpler models on Spambase.
- **Random Forest with tuned hyperparameters** is the most suitable choice when minimizing false positives is critical.
- Gradient Boosting is highly competitive out of the box but did not benefit from tuning here.
- Evaluating only on accuracy is misleading — Naive Bayes scores 83% accuracy yet would lose 136 legitimate emails per 921.
- Confusion matrix analysis (FP vs FN) is essential for any real-world filtering system.

---

## Future Work

- Apply NLP techniques (word embeddings, transformer-based models) on raw email text.
- Try modern gradient boosting libraries: **XGBoost**, **LightGBM**, **CatBoost**.
- Test on larger and more recent spam corpora.
- Deploy as a real-time filtering service and evaluate on live email traffic.

---

## References

1. Cranor, L. F., & LaMacchia, B. A. (1998). *Spam!*. Communications of the ACM.
2. UCI Machine Learning Repository — Spambase Dataset.
3. Analytics Vidhya — *Hyperparameter Tuning with GridSearchCV*. <https://www.analyticsvidhya.com/blog/2021/05/hyperparameter-tuning-gridsearchcv-randomizedsearchcv/>
4. Google Developers — *Machine Learning Crash Course: Classification*. <https://developers.google.com/machine-learning/crash-course/classification>
5. Course notes and slides, M.Sc. *Information Systems and Services*, University of Piraeus.
