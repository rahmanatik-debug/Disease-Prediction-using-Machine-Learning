# Model Evaluation Results

## Dataset
Breast Cancer Wisconsin Diagnostic Dataset

## Models Compared
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

## Evaluation Metrics

| Algorithm | Accuracy | Precision | Recall/Sensitivity | Specificity | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9762 | 0.9861 | 0.9954 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9286 | 0.9655 | 0.9932 |
| SVM | 0.9825 | 0.9861 | 0.9861 | 0.9762 | 0.9861 | 0.9950 |

## Best Performing Models

Based on the current 80:20 test split:

- **Logistic Regression**: Accuracy = 98.25%, F1-Score = 98.61%, ROC-AUC = 0.9954
- **SVM**: Accuracy = 98.25%, F1-Score = 98.61%, ROC-AUC = 0.9950
- **Random Forest**: Accuracy = 95.61%, F1-Score = 96.55%, ROC-AUC = 0.9932

Logistic Regression and SVM achieved the same Accuracy, Precision, Recall, Specificity and F1-Score on this test split. Logistic Regression has a slightly higher ROC-AUC.

## Conclusion

Logistic Regression produced the strongest overall result in this experiment, while SVM performed almost identically. Random Forest also performed well but had lower Accuracy and Specificity on this test split.

For medical prediction tasks, Accuracy should not be considered alone. Recall/Sensitivity, Specificity, F1-Score and ROC-AUC are also important when evaluating a classifier.

