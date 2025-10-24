#hala

# First, the model might perform well on training data by memorizing patterns specific to that data,
# but fail to generalize to new, unseen data. A test set provides an unbiased evaluation of the model's true performance.
# Also, the test set performance metrics (like R², accuracy, or mean IOU) give us a realistic
# expectation of how the model will perform when deployed.

import pandas as pd
import json
import numpy as np
from tqdm import tqdm


import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
from tqdm import tqdm

def intersection_area(box1, box2):
    # Kept for reference (not used in optimized path)
    x_left = max(box1['min_r'], box2['min_r'])
    y_bottom = max(box1['min_c'], box2['min_c'])
    x_right = min(box1['max_r'], box2['max_r'])
    y_top = min(box1['max_c'], box2['max_c'])

    if x_right < x_left or y_top < y_bottom:
        return 0
    else:
        return (x_right - x_left) * (y_top - y_bottom)

def union_area(box1, box2):
    # Kept for reference (not used in optimized path)
    area1 = (box1['max_r'] - box1['min_r']) * (box1['max_c'] - box1['min_c'])
    area2 = (box2['max_r'] - box2['min_r']) * (box2['max_c'] - box2['min_c'])
    return area1 + area2 - intersection_area(box1, box2)

def calculate_iou_optimized(file1_path, file2_path):
    usecols = ['filename', 'min_r', 'min_c', 'max_r', 'max_c']
    df1 = pd.read_excel(file1_path, usecols=usecols)
    df2 = pd.read_excel(file2_path, usecols=usecols)

    # Initialize IoU column (np.nan for unpaired rows)
    df1['iou'] = np.nan

    # Create pair_id based on row order in each file (replicates your iloc[idx] pairing)
    df1_for_merge = df1.reset_index()  # Preserves original index in 'index' column
    df1_for_merge['pair_id'] = df1_for_merge.groupby('filename').cumcount()

    df2_for_merge = df2[usecols].copy()
    df2_for_merge['pair_id'] = df2_for_merge.groupby('filename').cumcount()

    # Merge to pair corresponding rows (inner join handles min_rows automatically)
    merged = pd.merge(
        df1_for_merge,
        df2_for_merge,
        on=['filename', 'pair_id'],
        suffixes=('_1', '_2'),
        how='inner'
    )

    # Vectorized IoU calculation on all paired boxes at once
    if not merged.empty:
        # Extract box coordinates as NumPy arrays
        min_r1 = merged['min_r_1'].values
        min_c1 = merged['min_c_1'].values
        max_r1 = merged['max_r_1'].values
        max_c1 = merged['max_c_1'].values
        min_r2 = merged['min_r_2'].values
        min_c2 = merged['min_c_2'].values
        max_r2 = merged['max_r_2'].values
        max_c2 = merged['max_c_2'].values

        # Intersection
        x_left = np.maximum(min_r1, min_r2)
        y_bottom = np.maximum(min_c1, min_c2)
        x_right = np.minimum(max_r1, max_r2)
        y_top = np.minimum(max_c1, max_c2)
        inter_width = np.maximum(0, x_right - x_left)
        inter_height = np.maximum(0, y_top - y_bottom)
        inter = inter_width * inter_height

        # Union
        area1 = (max_r1 - min_r1) * (max_c1 - min_c1)
        area2 = (max_r2 - min_r2) * (max_c2 - min_c2)
        union = area1 + area2 - inter

        # IoU (avoid div-by-zero)
        iou = np.zeros_like(inter, dtype=float)
        mask = union > 0
        iou[mask] = inter[mask] / union[mask]

        # Assign back to original df1 using preserved indices
        df1.loc[merged['index'], 'iou'] = iou

    return df1


def get_and_write_results():
    iou_results = calculate_iou_optimized('training.xlsx', 'predictions_training.xlsx')
    print(f"Calculated {len(iou_results)} IoU values")
    print(iou_results.head(25))
    iou_results.to_csv("output.csv", index=False)

def mean_IOU(input):
    return np.average(input)

df = pd.read_csv("output.csv")

#  Logistic model on IOU
# Goal: use IoU (X) to predict category (y: 0 = not infected, 1 = infected)
# Steps:
#   1) Read labels from training.xlsx and merge with output.csv IoU by matching the box coordinates.
#   2) Fit a simple logistic regression: y ~ IOU.
#   3) Report R^2 (and accuracy for extra intuition).

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

def load_iou_with_labels(iou_csv_path: str, training_xlsx_path: str) -> pd.DataFrame:
    """
    Merge the IoU values with the true labels (category) from training.xlsx.
    We join on the 5 keys that uniquely describe a box in both files:
    filename, min_r, min_c, max_r, max_c.
    """
    # IoU that Katie saved
    iou_df = pd.read_csv(iou_csv_path)

    # Training file has the category  + box coordinates
    # We only load what we need to keep things simple.
    gt = pd.read_excel(
        "training.xlsx",
        usecols=["filename", "category", "min_r", "min_c", "max_r", "max_c"]
    )

    # Merge: left join so we only keep boxes for which we have IoU
    merged = pd.merge(
        iou_df,
        gt,
        on=["filename", "min_r", "min_c", "max_r", "max_c"],
        how="left",
        validate="1:1"
    )

    # Drop rows without a category (shouldn't happen if files align, but just in case)
    merged = merged.dropna(subset=["category", "iou"]).reset_index(drop=True)

    return merged

def fit_logistic_on_iou(data: pd.DataFrame):
    """
    Fit a logistic regression that predicts category from a single feature: IoU.
    Returns the fitted model, R^2 on training data, and accuracy on training data.
    """
    # X must be 2D for scikit-learn (n_samples, n_features).
    X = data["iou"].to_numpy().reshape(-1, 1)
    y = data["category"].astype(int).to_numpy()

    # Basic logistic regression (no fancy options).
    model = LogisticRegression()
    model.fit(X, y)

    # Predictions as probabilities (for R^2) and as class labels (for accuracy).
    y_prob = model.predict_proba(X)[:, 1]  # probability of class 1
    y_pred = model.predict(X)

    # R^2 between true labels and predicted probabilities.
    r2 = r2_score(y, y_prob)

    # Accuracy is also helpful for a classification task.
    acc = accuracy_score(y, y_pred)

    return model, r2, acc

# df (IoU only) was already read above in Katie code: df = pd.read_csv("output.csv")
# Merge it with labels from training.xlsx
iou_with_labels = load_iou_with_labels("output.csv", "training.xlsx")

# Fit logistic regression on the merged data
logit_model, r2_train, acc_train = fit_logistic_on_iou(iou_with_labels)

# Print short, clear results for the report
print(" Logistic regression on IOU")
print(f"Samples used: {len(iou_with_labels)}")
print(f"Train R^2 (y vs. predicted probability): {r2_train:.4f}")
print(f"Train Accuracy: {acc_train:.4f}")
