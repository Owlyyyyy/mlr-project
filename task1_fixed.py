"""
Part I: Classification Analysis (Heart.csv)
ECONM0011 Machine Learning for Economics

Updated version:
- Reports summary statistics for raw data
- Drops missing observations
- Determines interpretation of Sex from heart disease rates
- One-hot encodes categorical variables
- Standardises all non-categorical continuous variables:
  Age, RestBP, Chol, MaxHR, Oldpeak, Ca
- Reports summary statistics after preprocessing
- Uses stratified train/test split for held-out evaluation
- Tunes three classifiers on the training set only using StratifiedKFold:
  KNN, LDA, Logistic Regression
- Compares confusion matrices, ROC curves, AUCs, and test accuracies
- Predicts heart disease status for the specified patient using the best classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

print("=" * 60)
print("Q1: Summary Statistics of Raw Data")
print("=" * 60)

df = pd.read_csv("Heart.csv")

print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print("\nSummary statistics (raw data):")
print(df.describe(include="all"))

print("\n" + "=" * 60)
print("Q2: Preprocessing")
print("=" * 60)

df = df.dropna().copy()
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\nSex vs AHD rate:")
sex_ahd = df.groupby("Sex")["AHD"].apply(lambda x: (x == "Yes").mean())
print(sex_ahd)
print("\nInterpretation: the Sex value with the higher heart disease rate is treated as Male.")

df = pd.get_dummies(df, columns=["ChestPain", "Thal"], drop_first=True)
df["AHD"] = df["AHD"].map({"No": 0, "Yes": 1})

num_cols = ["Age", "RestBP", "Chol", "MaxHR", "Oldpeak", "Ca"]

scaler_report = StandardScaler()
df_report = df.copy()
df_report[num_cols] = scaler_report.fit_transform(df_report[num_cols])

print("\nContinuous variables standardised:")
print(num_cols)

print("\n" + "=" * 60)
print("Q2a: Summary Statistics After Preprocessing")
print("=" * 60)
print(df_report.describe())
print("\nNote: the standardised continuous variables should have mean ≈ 0 and std ≈ 1.")

print("\n" + "=" * 60)
print("Q3: Classification")
print("=" * 60)

x = df.drop(columns=["AHD"])
y = df["AHD"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(x_train)} samples")
print(f"Test set:  {len(x_test)} samples")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ("scale_num", StandardScaler(), num_cols),
    ],
    remainder="passthrough"
)

results = {}
fitted_models = {}

print("\n--- Classifier 1: KNN ---")

k_values = range(1, 31)
mean_scores = []

for k in k_values:
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(pipe, x_train, y_train, cv=skf, scoring="accuracy")
    mean_scores.append(scores.mean())

best_k = k_values[np.argmax(mean_scores)]
print(f"Best k: {best_k} (CV accuracy: {max(mean_scores):.4f})")
print(f"Optimal because k={best_k} gives the highest mean 5-fold stratified CV accuracy.")

knn_best = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=best_k))
])
knn_best.fit(x_train, y_train)
y_pred_knn = knn_best.predict(x_test)
y_prob_knn = knn_best.predict_proba(x_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)

results["KNN"] = {
    "accuracy": accuracy_score(y_test, y_pred_knn),
    "cm": confusion_matrix(y_test, y_pred_knn),
    "fpr": fpr_knn,
    "tpr": tpr_knn,
    "auc": roc_auc_score(y_test, y_prob_knn),
}
fitted_models["KNN"] = knn_best
print(f"Test Accuracy: {results['KNN']['accuracy']:.4f}")
print(f"Test AUC: {results['KNN']['auc']:.4f}")

print("\n--- Classifier 2: LDA ---")

param_grid = [
    {"solver": "svd", "shrinkage": None},
    {"solver": "lsqr", "shrinkage": None},
    {"solver": "lsqr", "shrinkage": "auto"},
    {"solver": "lsqr", "shrinkage": 0.1},
    {"solver": "lsqr", "shrinkage": 0.5},
    {"solver": "eigen", "shrinkage": None},
    {"solver": "eigen", "shrinkage": "auto"},
    {"solver": "eigen", "shrinkage": 0.1},
    {"solver": "eigen", "shrinkage": 0.5},
]

mean_scores = []
for params in param_grid:
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LinearDiscriminantAnalysis(
            solver=params["solver"], shrinkage=params["shrinkage"]
        ))
    ])
    scores = cross_val_score(pipe, x_train, y_train, cv=skf, scoring="accuracy")
    mean_scores.append(scores.mean())

best_index = int(np.argmax(mean_scores))
best_params = param_grid[best_index]
print(f"Best parameters: {best_params} (CV accuracy: {mean_scores[best_index]:.4f})")
print("Optimal because this solver/shrinkage combination gives the highest mean 5-fold stratified CV accuracy.")

lda_best = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearDiscriminantAnalysis(
        solver=best_params["solver"], shrinkage=best_params["shrinkage"]
    ))
])
lda_best.fit(x_train, y_train)
y_pred_lda = lda_best.predict(x_test)
y_prob_lda = lda_best.predict_proba(x_test)[:, 1]
fpr_lda, tpr_lda, _ = roc_curve(y_test, y_prob_lda)

results["LDA"] = {
    "accuracy": accuracy_score(y_test, y_pred_lda),
    "cm": confusion_matrix(y_test, y_pred_lda),
    "fpr": fpr_lda,
    "tpr": tpr_lda,
    "auc": roc_auc_score(y_test, y_prob_lda),
}
fitted_models["LDA"] = lda_best
print(f"Test Accuracy: {results['LDA']['accuracy']:.4f}")
print(f"Test AUC: {results['LDA']['auc']:.4f}")

print("\n--- Classifier 3: Logistic Regression ---")

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
mean_scores = []

for C in C_values:
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(C=C, max_iter=5000))
    ])
    scores = cross_val_score(pipe, x_train, y_train, cv=skf, scoring="accuracy")
    mean_scores.append(scores.mean())

best_C = C_values[np.argmax(mean_scores)]
print(f"Best C: {best_C} (CV accuracy: {max(mean_scores):.4f})")
print(f"Optimal because C={best_C} gives the highest mean 5-fold stratified CV accuracy.")

logit_best = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=best_C, max_iter=5000))
])
logit_best.fit(x_train, y_train)
y_pred_logit = logit_best.predict(x_test)
y_prob_logit = logit_best.predict_proba(x_test)[:, 1]
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_prob_logit)

results["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, y_pred_logit),
    "cm": confusion_matrix(y_test, y_pred_logit),
    "fpr": fpr_logit,
    "tpr": tpr_logit,
    "auc": roc_auc_score(y_test, y_prob_logit),
}
fitted_models["Logistic Regression"] = logit_best
print(f"Test Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
print(f"Test AUC: {results['Logistic Regression']['auc']:.4f}")

print("\n" + "=" * 60)
print("Q3b: Confusion Matrix Comparison")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = res["cm"]
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

for name, res in results.items():
    tn, fp, fn, tp = res["cm"].ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    print(f"\n{name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

print("\n" + "=" * 60)
print("Q3c: ROC Curve Comparison")
print("=" * 60)

plt.figure(figsize=(7, 6))
colors = ["#378ADD", "#D85A30", "#1D9E75"]
for (name, res), color in zip(results.items(), colors):
    plt.plot(res["fpr"], res["tpr"], color=color, linewidth=2,
             label=f"{name} (AUC = {res['auc']:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison — All Three Classifiers")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("Q3d: Accuracy Comparison")
print("=" * 60)

print(f"\n{'Method':<25} {'Test Accuracy':<15} {'AUC':<10}")
print("-" * 50)

best_method = None
best_acc = -np.inf
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:<15.4f} {res['auc']:<10.3f}")
    if res["accuracy"] > best_acc:
        best_acc = res["accuracy"]
        best_method = name

print(f"\nBest classifier: {best_method} with test accuracy {best_acc:.4f}")

plt.figure(figsize=(7, 4))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
bars = plt.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)
plt.ylabel("Test Accuracy")
plt.title("Accuracy Comparison Across Classifiers")
plt.ylim(max(0.0, min(accs) - 0.1), 1.0)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{acc:.4f}", ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("Q4: Patient Prediction")
print("=" * 60)

patient_raw = {
    "Age": 55,
    "Sex": 0,
    "RestBP": 130,
    "Chol": 246,
    "Fbs": 0,
    "RestECG": 2,
    "MaxHR": 150,
    "ExAng": 1,
    "Oldpeak": 1,
    "Slope": 2,
    "Ca": 0,
    "ChestPain_nonanginal": 0,
    "ChestPain_nontypical": 0,
    "ChestPain_typical": 1,
    "Thal_normal": 1,
    "Thal_reversable": 0,
}

patient_df = pd.DataFrame([patient_raw])

for col in x_train.columns:
    if col not in patient_df.columns:
        patient_df[col] = 0

patient_df = patient_df[x_train.columns]

print("\nPatient features (raw before pipeline preprocessing):")
print(patient_df.to_string(index=False))

best_model = fitted_models[best_method]
prediction = best_model.predict(patient_df)[0]
probability = best_model.predict_proba(patient_df)[0]

print(f"\nUsing best classifier: {best_method}")
print(f"Prediction: {'Heart Disease (Yes)' if prediction == 1 else 'No Heart Disease (No)'}")
print(f"Probability of heart disease: {probability[1]:.4f}")
print(f"Probability of no heart disease: {probability[0]:.4f}")

