import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

rows, predictions, scores = [], {}, {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    predictions[name], scores[name] = pred, score

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall/Sensitivity": recall_score(y_test, pred),
        "Specificity": tn / (tn + fp),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, score)
    })

results = pd.DataFrame(rows)
results.to_csv("results/model_metrics.csv", index=False)

metrics = ["Accuracy", "Precision", "Recall/Sensitivity", "Specificity", "F1-Score", "ROC-AUC"]
ax = results.set_index("Algorithm")[metrics].plot(kind="bar", figsize=(14, 7), width=0.78)
plt.title("Performance Comparison of Machine Learning Algorithms", fontsize=18, fontweight="bold")
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Score")
plt.ylim(0, 1.08)
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metrics", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.3)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, pred) in zip(axes, predictions.items()):
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
fig.suptitle("Confusion Matrices", fontsize=17, fontweight="bold")
plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 7))
for name, score in scores.items():
    fpr, tpr, _ = roc_curve(y_test, score)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc_score(y_test, score):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.title("ROC Curve Comparison", fontsize=17, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/roc_curve_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

print(results.round(4))
