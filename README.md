# Breast Cancer Prediction Using Machine Learning

## Overview
A complete machine-learning classification project using the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn.

### Models
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

### Evaluation Metrics
- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-Score
- ROC-AUC

> Educational use only. This project is not a clinical diagnostic system.

## Dataset
The dataset is loaded directly with `sklearn.datasets.load_breast_cancer`.

- Samples: 569
- Numerical features: 30
- Classes: malignant and benign

## Preprocessing
1. Load and inspect the dataset
2. Check missing values
3. Check duplicates
4. Separate features and target
5. 80:20 stratified train-test split
6. Standardize features using `StandardScaler`

## Models
### Logistic Regression
Used as a baseline linear classifier.

### Random Forest
An ensemble of decision trees for nonlinear classification.

### SVM
A margin-based classifier suitable for high-dimensional numerical data.

## Evaluation
Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score and ROC-AUC are calculated. Confusion matrices and ROC curves are also generated.

## Repository Structure
```text
Breast_Cancer_ML_GitHub_Repository/
├── Breast_Cancer_ML_Project.ipynb
├── README.md
├── requirements.txt
└── results/
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── roc_curve_comparison.png
    └── model_metrics.csv
```

## How to Run
```bash
pip install -r requirements.txt
Breast_Cancer_ML_Project (1).ipynb
```



## Conclusion
The project demonstrates a reproducible workflow for disease classification, including preprocessing, model training, evaluation, and comparison. For medical prediction, model selection should consider
Recall/Sensitivity, Specificity, F1-Score and ROC-AUC in addition to Accuracy.
The project demonstrates a reproducible workflow for disease classification, including preprocessing, model training, evaluation, and comparison. For medical prediction, model selection should consider Recall/Sensitivity, Specificity, F1-Score and ROC-AUC in addition to Accuracy.

