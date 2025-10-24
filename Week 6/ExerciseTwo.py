#I have used these variables names:X_train, X_test, y_train, y_test, model



# Plot and Select appropriate model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("winequality-red.csv", sep=";")


def analyze_and_clean_dataset(df):
    print(f"Dataset shape: {df.shape}")

    # 1) Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\nNo missing values")
    else:
        print("\nMissing values:")
        print(missing.to_string())

    # 2) Duplicates
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}")
    df = df.drop_duplicates()
    # 3) Data types
    print("\nData types:")
    print(df.dtypes.to_string())


    # 5) Outlier analysis (IQR) for numeric columns
    cols = df.columns

    df = df.apply(pd.to_numeric, errors='coerce', axis=1)

    n = len(df)
    rows = []
    for col in cols:
        x = df[col].dropna()
        if x.empty:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lb) | (df[col] > ub)
        cnt = int(mask.sum())
        if cnt:
            rows.append({
                "column": col,
                "outliers": cnt,
                "percent": round(100 * cnt / n, 2)
            })
    if rows:
        print("\nOutlier summary:")
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("\nNo notable outliers")
    return df

df = analyze_and_clean_dataset(df)
# Outliers have been noted, but are not treated here
"""
Step 0 — Get the data

If Katie already produced a cleaned DataFrame called `df`, reuse it.
Otherwise, load the raw CSV so this file still runs on its own.
We also keep only numeric columns (wine dataset is numeric anyway).
"""

try:
    df  # reuse Katie df if defined
except NameError:
    df = pd.read_csv("winequality-red.csv")

df = df.select_dtypes(include=[np.number])
print("boo")
"""
Step 1 
1a) Target distribution: histogram of 'quality'
1b) A few scatter plots: feature vs. target, to eyeball simple trends
1c) Numerical correlations with 'quality' (top values)
"""

# 1a) Quality distribution
plt.figure()
df["quality"].plot(kind="hist", bins=12, edgecolor="black")
plt.title("Quality distribution")
plt.xlabel("quality")
plt.tight_layout()
plt.show()

# 1b) A few simple feature vs target plots

for col in ["alcohol", "sulphates", "volatile acidity", "citric acid"]:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="quality", y=col, data=df)
        plt.title(f"{col} vs Quality")
        plt.tight_layout()
        plt.show()

# 1c) Top linear correlations with target (numbers only)
corr_to_quality = (
    df.corr(numeric_only=True)["quality"]
      .drop(labels=["quality"])
      .sort_values(ascending=False)
)
# Keep this in a variable; so teammates can print or inspect in notebook if they want
top_corr_with_quality = corr_to_quality.head(6)

"""
Step 2 — Train/Test split

Target (y) is 'quality'; features (X) are the remaining columns.
Use stratified split so the distribution of quality is similar in train and test.
"""

from sklearn.model_selection import train_test_split

y = df["quality"].astype(float)
X = df.drop(columns=["quality"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

"""
Step 3 — Simple baseline model (training performance only)

Fit a plain Linear Regression on the TRAINING set.
We report only TRAINING metrics here (R² and RMSE).
Hal will evaluate on the TEST set and try stronger models.
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

# Training-only performance (leave test evaluation to Person C)
y_pred_train = model.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)
rmse_train = mean_squared_error(y_train, y_pred_train)

# Variables exposed for Hala:
# - X_train, X_test, y_train, y_test
# - model (fitted on training data)

#hala
# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Initialize and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate performance
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print("Random Forest Performance:")
print(f"R² Score: {rf_r2:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"MAE: {rf_mae:.4f}")

#try a second model: Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

# Initialize and train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluate performance
gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_mae = mean_absolute_error(y_test, y_pred_gb)

print("\nGradient Boosting Performance:")
print(f"R² Score: {gb_r2:.4f}")
print(f"RMSE: {gb_rmse:.4f}")
print(f"MAE: {gb_mae:.4f}")

# Compare models
# Create comparison table
comparison = {
    'Model': ['Random Forest', 'Gradient Boosting'],
    'R² Score': [rf_r2, gb_r2],
    'RMSE': [rf_rmse, gb_rmse],
    'MAE': [rf_mae, gb_mae]
}

import pandas as pd
comparison_df = pd.DataFrame(comparison)
print("\nModel Comparison:")
print(comparison_df)

# Determine best model
if rf_r2 > gb_r2:
    best_model = "Random Forest"
    best_r2 = rf_r2
else:
    best_model = "Gradient Boosting"
    best_r2 = gb_r2

print(f"\nBest Performing Model: {best_model} (R² = {best_r2:.4f})")



# Analyze feature importance for the best model
if best_model == "Random Forest":
    feature_importance = rf_model.feature_importances_
else:
    feature_importance = gb_model.feature_importances_

features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Visualize feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Wine Quality Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#Limitations:
#Ordinal target treated as continuous - loses rating nature

#Imbalanced classes - biases toward average scores

#Subjective quality ratings - human bias in labels

#Basic features - no transformations or interactions

#Small dataset - 1599 samples limits complex models

#Improvements:
#Better models: Ordinal regression, classification, or neural networks

#Feature engineering: Interactions, polynomials, scaling

#Advanced techniques: Cross-validation, ensembles, Bayesian optimization

#More data: Additional samples and features (region, grape type)

#Model interpretation: SHAP values, partial dependence plots