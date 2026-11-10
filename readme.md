# Task 4: Classification with Logistic Regression

## Objective
Build a binary classifier using logistic regression to detect malignant vs benign tumours.

## Dataset
Breast Cancer Wisconsin dataset (569 samples, 30 numeric features). Target: diagnosis (M = malignant = 1, B = benign = 0).

## Steps Performed
1. **Preprocessing:** dropped id and the empty column, encoded the diagnosis as 0/1.
2. **Train-test split:** 80/20, stratified, then standardized features (scaler fitted on the training data only).
3. **Model:** fitted LogisticRegression from scikit-learn.
4. **Evaluation:** confusion matrix, precision, recall and ROC-AUC.
5. **Threshold tuning and sigmoid:** compared thresholds from 0.2 to 0.8, plotted the sigmoid function and explained it.

## Results (threshold 0.5)
| Metric | Value |
|--------|-------|
| Accuracy | 0.965 |
| Precision | 0.975 |
| Recall | 0.929 |
| ROC-AUC | 0.996 |

## Threshold Tuning
Lowering the threshold to 0.4 reduced missed malignant cases from 3 to 2 (recall about 0.95) with precision about 0.98, so 0.4 was chosen because missing a malignant case is more costly than a false alarm.

## Key Findings
- The model separates the classes very well (ROC-AUC 0.996).
- The strongest features pushing towards malignant are texture_worst, radius_se and symmetry_worst.
- The sigmoid function turns the linear score into a probability between 0 and 1.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Files
- `logistic_regression.ipynb`: notebook with all steps
- `data.csv`: dataset
- `Screenshots/`: key outputs and plots