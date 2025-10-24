#I have used these variable names for part A:X_train, X_test, y_train, y_test
# For part B:model (the trained LinearRegression model), X_train, y_train

# IMPORTS
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


np.random.seed(2)

x = np.random.uniform(0, 10, 200)
y = 2*x**2 - 5 *x + 3 + np.random.normal(0, 10, 200)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset")
plt.show()

# Decision regression model, it looks either like an exponential growing, or like the RHS of a parabola

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, train_size=0.8)

input_train = pd.DataFrame({"x": X_train, "y": y_train})
input_test = pd.DataFrame({"x": X_test, "y": y_test})

train_x = input_train["x"]
train_x_sq = input_train["x"]**2
train_x_poly = pd.DataFrame({"x": train_x, "x_sq": train_x_sq})
X_poly = sm.add_constant(train_x_poly)

# Fit the model
model = sm.OLS(input_train["y"], X_poly).fit()


# Train and evaluate on training set
def build_design_matrix_with_intercept(x: np.ndarray) -> np.ndarray:
    """
    Make the design matrix X with a leading column of ones (intercept).
    Input x can be shape (n,) or (n,1). Output X is (n,2): [1, x].
    """
    x = np.asarray(x).reshape(-1, 1)        # ensure column vector
    ones = np.ones((x.shape[0], 1))
    X = np.hstack([ones, x])                # [1, x]
    return X

def fit_ols_betas(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute OLS coefficients beta_hat = (X^T X)^(-1) X^T y.
    Uses lstsq (more stable than forming an explicit inverse).
    Returns array [b0, b1].
    """
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta_hat  # [b0, b1]

def predict_linear(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute y_hat = X @ beta."""
    return X @ beta

def r2_and_mse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    R^2 = 1 - SSE/SST,  MSE = SSE / n
    """
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - (sse / sst)
    mse = sse / y_true.size
    return r2, mse

def train_and_evaluate_on_training(X_train, y_train):
    """
    Workflow:
    1) Build design matrix with intercept
    2) Fit OLS to get b0, b1
    3) Predict on training
    4) Report R^2 and MSE on training
    Returns: dict with b0, b1, y_hat_train, r2_train, mse_train
    """
    # 1) design matrix
    Xtr = build_design_matrix_with_intercept(X_train)

    # 2) fit OLS (beta_hat = [b0, b1])
    beta_hat = fit_ols_betas(Xtr, y_train)
    b0, b1 = float(beta_hat[0]), float(beta_hat[1])

    # 3) predict on training
    y_hat_tr = predict_linear(Xtr, beta_hat)

    # 4) training metrics
    r2_tr, mse_tr = r2_and_mse(y_train, y_hat_tr)

    # Print a short, readable summary (Possibly help for hala later)
    print("TRAINED OLS (training set) ")
    print(f"Model:  y_hat = b0 + b1·x  with  b0 = {b0:.4f},  b1 = {b1:.4f}")
    print(f"Training R² = {r2_tr:.4f}")
    print(f"Training MSE = {mse_tr:.4f}")

    return {
        "b0": b0,
        "b1": b1,
        "y_hat_train": y_hat_tr,
        "r2_train": r2_tr,
        "mse_train": mse_tr,
    }

# Hala: Evaluate on test data and refine/iterate

test_x = input_test["x"]
test_x_sq = input_test["x"]**2
test_x_poly = pd.DataFrame({"x": test_x, "x_sq": test_x_sq})
X_test_poly = sm.add_constant(test_x_poly)

# Evaluate initial model on test data
print("=== INITIAL MODEL EVALUATION (Linear Regression) ===")
y_test_pred_linear = model.predict(X_test_poly)
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

# Reshape X_train and X_test for sklearn (they need to be 2D)
X_train_2d = X_train.reshape(-1, 1)
X_test_2d = X_test.reshape(-1, 1)

# Train polynomial model
poly_model.fit(X_train_2d, y_train)

# Evaluate polynomial model on test data
y_test_pred_poly = poly_model.predict(X_test_2d)
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
print("As expected, simple linear regression performs worse because it cannot capture the quadratic relationship.")
