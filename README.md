# Stroke Risk Classification using PySpark

Four-model comparison for stroke risk prediction with PySpark MLlib.

## Introduction
Individual educational project exploring a classification pipeline on highly imbalanced healthcare data, where only about 4.9% of records are stroke cases. Uses the [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), containing 5,110 rows and 10 features.

Compares Random Forest, Gradient Boosting, SVM (LinearSVC), and MLP. Random Forest and Gradient Boosting are tuned with `CrossValidator`, while SVM and MLP use manually set parameters. Class imbalance is handled with SMOTE on the training set only, combined with a 0.30 classification threshold.

> **Disclaimer:** For educational purposes only, not a medical diagnostic tool.

## Results
All metrics are computed on the original, non-resampled test set (987 rows, 55 stroke cases) using a 0.30 probability threshold. SVM uses its default prediction.

**Before tuning**
| Model | AUC-ROC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Random Forest | 78.11% | 77.81% | 13.72% | 56.36% | 22.06% |
| Gradient Boosting | 71.29% | 87.74% | 17.00% | 30.91% | 21.94% |
| SVM | 78.93% | 69.00% | 11.38% | 67.27% | 19.47% |
| MLP | 76.05% | 67.07% | 11.65% | 74.55% | 20.15% |

**After tuning (Random Forest and Gradient Boosting only)**
| Model | AUC-ROC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Tuned Random Forest | 78.11% | 77.81% | 13.72% | 56.36% | 22.06% |
| Tuned Gradient Boosting | 70.32% | 86.73% | 17.24% | 36.36% | 23.39% |

**Tuning details**
- **Random Forest:** 3-fold CrossValidator optimizing AUC-ROC.
- **Gradient Boosting:** 2-fold CrossValidator optimizing AUC-ROC. Grid: `maxIter` [20, 40] × `maxDepth` [3, 5] × `stepSize` [0.05, 0.1].

**Key takeaways**
- Tuning Random Forest produced test metrics identical to the manual configuration, with no measurable improvement.
- Tuning Gradient Boosting raised recall (30.91% → 36.36%) and F1 (21.94% → 23.39%) while lowering AUC-ROC (71.29% → 70.32%). The recall gain amounts to roughly 3 additional stroke cases out of 55.
- SVM achieved the highest AUC-ROC (78.93%), MLP the highest recall (74.55%), and Gradient Boosting the highest accuracy and precision.
- Tuned Gradient Boosting achieved the best F1 (23.39%), though all F1 scores fall within a narrow range (19–23%).
- Precision remains low across all models (11–17%), producing many false positives. In medical screening, this trades off against higher recall.

## Contributor Expectations
Individual learning project, so pull requests are not accepted.

## Known Issues
- Only Random Forest and Gradient Boosting are tuned, so the comparison with SVM and MLP is not fully equal.
- `CrossValidator` runs 2 folds (GB) and 3 folds (RF) on SMOTE-resampled data, so CV scores tend to be optimistic and less stable.
- SMOTE is applied after encoding, so categorical features can take fractional values. `SMOTENC` or class weights are possible alternatives.
- BMI imputation is computed before the train/test split, causing minor data leakage from the test set.
- The 0.30 threshold does not apply to SVM because `LinearSVC` does not output probabilities.
- The test set contains only 55 stroke cases, so small differences between models may not be significant.
