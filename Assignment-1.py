"""
AI-ML Assignment 1
Medical Insurance Cost Prediction using Multiple Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================================
# TASK 1: DATA UNDERSTANDING (2 Marks)
# ============================================================
print("=" * 60)
print("TASK 1: DATA UNDERSTANDING")
print("=" * 60)

# 1. Load the dataset using Pandas
df = pd.read_csv("insurance.csv")

# 2. Display the first five records
print("\nFirst 5 records:")
print(df.head())

# 3. Identify numerical, categorical, and target features
numerical_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]
target_variable = "charges"

print(f"\nNumerical features   : {numerical_features}")
print(f"Categorical features : {categorical_features}")
print(f"Target variable      : {target_variable}")

print("\nDataset shape:", df.shape)
print("\nColumn data types:")
print(df.dtypes)


# ============================================================
# TASK 2: DATA PREPROCESSING (2 Marks)
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: DATA PREPROCESSING")
print("=" * 60)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Encode categorical variables (sex, smoker, region)
df_encoded = df.copy()

# Binary encoding for sex and smoker
df_encoded["sex"] = df_encoded["sex"].map({"male": 0, "female": 1})
df_encoded["smoker"] = df_encoded["smoker"].map({"no": 0, "yes": 1})

# One-hot encoding for region (has more than 2 categories)
df_encoded = pd.get_dummies(df_encoded, columns=["region"], drop_first=True)

print("\nEncoded dataset preview:")
print(df_encoded.head())

# Split features (X) and target (y)
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size : {X_test.shape[0]} rows")


# ============================================================
# TASK 3: MODEL DEVELOPMENT (3 Marks)
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: MODEL DEVELOPMENT")
print("=" * 60)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel trained successfully.")
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:12s}: {coef:.2f}")
print(f"  {'Intercept':12s}: {model.intercept_:.2f}")


# ============================================================
# TASK 4: MODEL EVALUATION (2 Marks)
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: MODEL EVALUATION")
print("=" * 60)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE) : {mae:.2f}")
print(f"Mean Squared Error (MSE)  : {mse:.2f}")
print(f"R2 Score                  : {r2:.4f}")

# Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="teal", edgecolor="k")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    label="Ideal fit",
)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
print("\nScatter plot saved as 'actual_vs_predicted.png'")

print("""
Observations:
1. The model achieves a reasonably high R2 score, meaning a large portion
   of the variance in insurance charges can be explained by age, sex, BMI,
   children, smoker status, and region.
2. Points lie close to the diagonal 'ideal fit' line for lower charge
   values, but the model under-predicts for customers with very high
   actual charges (mostly smokers), showing higher error at the extremes.
3. The 'smoker' feature has by far the largest coefficient magnitude,
   confirming it is the strongest driver of insurance cost in this model.
""")


# ============================================================
# TASK 5: CONCLUSION (1 Mark)
# ============================================================
print("=" * 60)
print("TASK 5: CONCLUSION")
print("=" * 60)

conclusion = f"""
This project built a Multiple Linear Regression model to predict medical
insurance charges using age, sex, BMI, number of children, smoking status,
and region. The model achieved an R2 score of {r2:.2f}, an MAE of {mae:.0f},
and an MSE of {mse:.0f} on the test set, indicating a reasonably good fit.
Smoking status emerged as the single strongest predictor of charges,
followed by age and BMI, while sex and region had comparatively minor
effects. These findings align with real-world expectations, since smokers
and older, higher-BMI individuals generally carry greater health risk and
therefore higher insurance costs. A key limitation of Linear Regression
here is its assumption of a linear relationship between features and
charges; in reality, costs rise sharply and non-linearly for high-risk
groups such as smokers, which the model cannot fully capture. A non-linear
approach (e.g. polynomial regression or tree-based models) may fit this
data better.
"""
print(conclusion)

with open("conclusion.txt", "w") as f:
    f.write(conclusion.strip())

print("\nAll tasks completed. Outputs saved: actual_vs_predicted.png, conclusion.txt")