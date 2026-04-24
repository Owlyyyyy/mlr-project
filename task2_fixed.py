"""
Part II: Regression Analysis (Credit.csv)
ECONM0011 Machine Learning for Economics

Updated version:
- Drops missing observations
- Drops ID column
- Creates dummy variables for Student, Married, Gender, and Ethnicity
- Reports summary statistics after preprocessing
- Standardises continuous variables before building the nonlinear design
- Constructs quadratic and interaction terms from the standardised continuous variables
  and characteristic dummies
- Uses 5-fold CV Lasso to choose lambda
- Plots lambda selection figure and coefficient path
- Predicts the specified person's balance using Lasso
- Fits a depth-3 decision tree and visualises it
- Compares random forest test MSE across required tree counts, with and without depth cap
- Predicts the specified person's balance using the best random forest
- Plots variable importance and compares with Lasso findings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

print("=" * 60)
print("Part II: Regression Analysis")
print("=" * 60)

df = pd.read_csv("Credit.csv")
print(f"\nRaw data shape: {df.shape}")
print("Raw summary statistics:")
print(df.describe(include="all"))

df = df.dropna().copy()
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

df["Student"] = df["Student"].str.strip()
df["Married"] = df["Married"].str.strip()
df["Gender"] = df["Gender"].str.strip()

df["Student_num"] = df["Student"].map({"No": 0, "Yes": 1})
df["Married_num"] = df["Married"].map({"No": 0, "Yes": 1})
df["Gender_num"] = df["Gender"].map({"Male": 0, "Female": 1})

df = pd.get_dummies(df, columns=["Ethnicity"], drop_first=True)
df = df.drop(columns=["Student", "Married", "Gender"])

print("\n" + "=" * 60)
print("Q1a: Summary Statistics After Preprocessing")
print("=" * 60)
print(df.describe())
print(f"\nColumns: {list(df.columns)}")

print("\n" + "=" * 60)
print("Q1b: Lasso Regression with 5-Fold CV")
print("=" * 60)

y = df["Balance"]
X_raw = df.drop(columns=["Balance"]).copy()
raw_columns = X_raw.columns.tolist()

cont_cols = ["Income", "Limit", "Rating", "Cards", "Age", "Education"]
dummy_cols = [c for c in X_raw.columns if c not in cont_cols]

print(f"\nContinuous vars: {cont_cols}")
print(f"Dummy vars: {dummy_cols}")

scaler_report = StandardScaler()
X_report = X_raw.copy()
X_report[cont_cols] = scaler_report.fit_transform(X_report[cont_cols])

poly_report = PolynomialFeatures(degree=2, include_bias=False)
X_poly_report = poly_report.fit_transform(X_report)
feature_names = poly_report.get_feature_names_out(X_report.columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("scale_cont", StandardScaler(), cont_cols),
        ("keep_dummy", "passthrough", dummy_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

lasso_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lasso_cv", LassoCV(alphas=np.logspace(-4, 2, 100), cv=5, random_state=42, max_iter=50000)),
])

lasso_pipeline.fit(X_raw, y)
lasso_cv = lasso_pipeline.named_steps["lasso_cv"]

best_lambda = lasso_cv.alpha_
best_coefficients = lasso_cv.coef_
best_intercept = lasso_cv.intercept_

print(f"\nBest lambda: {best_lambda:.6f}")
print(f"Intercept: {best_intercept:.3f}")
print(f"Non-zero coefficients: {np.sum(best_coefficients != 0)} / {len(best_coefficients)}")

print("\nQ1b-i: Lambda Selection Figure")

alphas = lasso_cv.alphas_
mean_mse = lasso_cv.mse_path_.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(alphas, mean_mse, linewidth=2)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel("Lambda (α)")
plt.ylabel("Mean Validation MSE")
plt.axvline(best_lambda, linestyle="--", color="red", label=f"Optimal λ = {best_lambda:.4f}")
plt.title("Lasso 5-Fold CV: Lambda Selection")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lasso_lambda_selection.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nQ1b-ii: Coefficient Path Plot")

coef_paths = []
alphas_path = np.logspace(-4, 2, 100)

for alpha in alphas_path:
    lasso_temp = Lasso(alpha=alpha, max_iter=50000)
    lasso_temp.fit(X_poly_report, y)
    coef_paths.append(lasso_temp.coef_.copy())

coef_paths = np.array(coef_paths)

plt.figure(figsize=(10, 6))
for i in range(coef_paths.shape[1]):
    plt.plot(alphas_path, coef_paths[:, i], linewidth=0.8)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel("Lambda (α)")
plt.ylabel("Coefficient Value")
plt.title("Lasso Coefficient Path (Figure 6.6 style)")
plt.axvline(best_lambda, linestyle="--", color="red", alpha=0.7, label=f"Optimal λ = {best_lambda:.4f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lasso_coefficient_path.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nNon-zero coefficients at optimal lambda:")
for name, coef in zip(feature_names, best_coefficients):
    if coef != 0:
        print(f"  {name}: {coef:.4f}")

print("\n" + "=" * 60)
print("Q1c: Lasso Prediction for Specified Person")
print("=" * 60)

person_data = {
    "Income": 100,
    "Limit": 6000,
    "Rating": 500,
    "Cards": 3,
    "Age": 70,
    "Education": 12,
    "Student_num": 0,
    "Married_num": 1,
    "Gender_num": 1,
    "Ethnicity_Asian": 1,
    "Ethnicity_Caucasian": 0,
}

for col in raw_columns:
    if col not in person_data:
        person_data[col] = 0

person_df = pd.DataFrame([person_data])[raw_columns]

print("\nPerson features:")
print(person_df.to_string(index=False))

predicted_balance_lasso = lasso_pipeline.predict(person_df)[0]
print(f"\nPredicted balance (Lasso): {predicted_balance_lasso:.2f}")

print("\n" + "=" * 60)
print("Q2: Tree-Based Methods")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42
)
n_trees_list = [1, 5, 10, 50, 100, 200]

print("\nQ2a: Decision Tree (max_depth=3)")

tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f"Test MSE (single tree): {mse_tree:.2f}")

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=list(X_raw.columns), filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree (max_depth=3)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()

root_feature = X_raw.columns[tree.tree_.feature[0]]
print("\nInterpretation:")
print(f"The root node splits on {root_feature}, making it the most influential predictor at the first split.")
print("Variables appearing closer to the root have greater predictive importance in this tree.")
print("The depth-3 tree gives an interpretable summary of the main drivers of credit card balance.")

print("\nQ2b: Random Forest MSEs")

mse_depth3 = []
for n in n_trees_list:
    rf = RandomForestRegressor(n_estimators=n, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    mse_depth3.append(mean_squared_error(y_test, rf.predict(X_test)))

mse_full = []
for n in n_trees_list:
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    mse_full.append(mean_squared_error(y_test, rf.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(n_trees_list, mse_depth3, marker="o", label="RF (max_depth=3)")
plt.plot(n_trees_list, mse_full, marker="o", label="RF (no depth limit)")
plt.axhline(y=mse_tree, color="r", linestyle="--", label=f"Single tree (MSE={mse_tree:.0f})")
plt.xlabel("Number of Trees")
plt.ylabel("Test MSE")
plt.title("Random Forest Test MSE Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rf_mse_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n{'n_trees':<10} {'MSE (depth=3)':<20} {'MSE (no limit)':<20}")
print("-" * 50)
for n, m3, mf in zip(n_trees_list, mse_depth3, mse_full):
    print(f"{n:<10} {m3:<20.2f} {mf:<20.2f}")
print(f"{'Single':<10} {mse_tree:<20.2f}")

print("\nDiscussion:")
print("As the number of trees increases, test MSE typically falls because averaging across trees reduces variance.")
print("The unrestricted forests usually achieve lower MSE than the depth-3 forests because they can capture more complex patterns.")
print("The gains become smaller as the forest gets larger, showing diminishing returns from adding more trees.")

print("\n" + "=" * 60)
print("Q2c: Random Forest Prediction for Specified Person")
print("=" * 60)

all_mses = mse_depth3 + mse_full
all_configs = [(n, 3) for n in n_trees_list] + [(n, None) for n in n_trees_list]
best_idx = int(np.argmin(all_mses))
best_n_trees, best_depth = all_configs[best_idx]
best_mse_val = all_mses[best_idx]

print(f"Best RF: n_trees={best_n_trees}, max_depth={best_depth}, MSE={best_mse_val:.2f}")

best_rf = RandomForestRegressor(
    n_estimators=best_n_trees,
    max_depth=best_depth,
    random_state=42
)
best_rf.fit(X_train, y_train)

predicted_balance_rf = best_rf.predict(person_df)[0]
print(f"Predicted balance (Random Forest): {predicted_balance_rf:.2f}")
print(f"\nComparison: Lasso predicted {predicted_balance_lasso:.2f}, RF predicted {predicted_balance_rf:.2f}")

print("\n" + "=" * 60)
print("Q2d: Variable Importance")
print("=" * 60)

importance_df = pd.DataFrame({
    "Feature": X_raw.columns,
    "Importance": best_rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nRandom Forest Variable Importance:")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Variable Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("rf_variable_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nComparison with Lasso:")
print("Lasso identifies important predictors by retaining non-zero coefficients, while random forests use split-based feature importance.")
print("If the same variables appear as important in both methods, that strengthens the evidence that those predictors matter for balance.")

lasso_main_effects = []
for name, coef in zip(feature_names, best_coefficients):
    if coef != 0 and " " not in name:
        lasso_main_effects.append((name, abs(coef)))
lasso_main_effects.sort(key=lambda x: x[1], reverse=True)

print("\nTop Lasso main effects (by |coefficient|):")
for name, coef in lasso_main_effects[:10]:
    print(f"  {name}: {coef:.4f}")
