#I have used these variable names for part A:X_train, X_test, y_train, y_test
# For part B:model (the trained LinearRegression model), X_train, y_train 





# Hala: Evaluate on test data and refine/iterate

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Evaluate initial model on test data
print("=== INITIAL MODEL EVALUATION (Linear Regression) ===")
y_test_pred_linear = model.predict(X_test)
test_r2_linear = r2_score(y_test, y_test_pred_linear)
test_mse_linear = mean_squared_error(y_test, y_test_pred_linear)

print(f"Test R²: {test_r2_linear:.4f}")
print(f"Test MSE: {test_mse_linear:.4f}")

# Since the data appears to have a quadratic relationship (based on the generation formula),
# let's try a polynomial regression as an alternative model

print("\n=== ALTERNATIVE MODEL (Polynomial Regression - Degree 2) ===")

# Create polynomial regression model
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Train polynomial model
poly_model.fit(X_train, y_train)

# Evaluate polynomial model on test data
y_test_pred_poly = poly_model.predict(X_test)
test_r2_poly = r2_score(y_test, y_test_pred_poly)
test_mse_poly = mean_squared_error(y_test, y_test_pred_poly)

print(f"Test R²: {test_r2_poly:.4f}")
print(f"Test MSE: {test_mse_poly:.4f}")

# Compare models
print("\n=== MODEL COMPARISON ===")
print(f"Linear Regression - Test R²: {test_r2_linear:.4f}, Test MSE: {test_mse_linear:.4f}")
print(f"Polynomial Regression (deg 2) - Test R²: {test_r2_poly:.4f}, Test MSE: {test_mse_poly:.4f}")

# Determine which model performs better
if test_r2_poly > test_r2_linear:
    print("\n✓ Polynomial regression performs better (higher R²)")
    best_model = poly_model
    best_model_name = "Polynomial Regression (degree 2)"
else:
    print("\n✓ Linear regression performs better (higher R²)")
    best_model = model
    best_model_name = "Linear Regression"

print(f"Recommended model: {best_model_name}")

# Explanation
print("\n=== EXPLANATION ===")
print("Based on the data generation formula (y = 2*x² - 5*x + 3 + noise),")
print("we expect a quadratic relationship. The polynomial regression (degree 2)")
print("should capture this relationship better than simple linear regression.")
print("This is because it can model the curvature in the data that linear")
print("regression cannot capture.")