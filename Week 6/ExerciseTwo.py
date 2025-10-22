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