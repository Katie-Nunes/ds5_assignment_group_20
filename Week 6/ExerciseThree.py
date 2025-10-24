#hala

# First, the model might perform well on training data by memorizing patterns specific to that data,
# but fail to generalize to new, unseen data. A test set provides an unbiased evaluation of the model's true performance.
# Also, the test set performance metrics (like R², accuracy, or mean IOU) give us a realistic
# expectation of how the model will perform when deployed.

import pandas as pd
import json
import numpy as np
from tqdm import tqdm


def intersection_area(box1, box2):
    x_left = max(box1['min_r'], box2['min_r'])
    y_bottom = max(box1['min_c'], box2['min_c'])
    x_right = min(box1['max_r'], box2['max_r'])
    y_top = min(box1['max_c'], box2['max_c'])

    if x_right < x_left or y_top < y_bottom:
        return 0
    else:
        return (x_right - x_left) * (y_top - y_bottom)

def union_area(box1, box2):
    area1 = (box1['max_r'] - box1['min_r']) * (box1['max_c'] - box1['min_c'])
    area2 = (box2['max_r'] - box2['min_r']) * (box2['max_c'] - box2['min_c'])
    return area1 + area2 - intersection_area(box1, box2)

def calculate_iou_optimized(file1_path, file2_path):
    # Load only the first 100 rows and necessary columns
    usecols = ['filename', 'min_r', 'min_c', 'max_r', 'max_c']
    df1 = pd.read_excel(file1_path, usecols=usecols).head(10)
    df2 = pd.read_excel(file2_path, usecols=usecols).head(10)

    # Add a new column to store IoU values
    df1['iou'] = None

    # Group by filename for faster lookup
    df1_grouped = df1.groupby('filename')
    df2_grouped = df2.groupby('filename')

    # Get common filenames
    common_filenames = set(df1['filename']).intersection(set(df2['filename']))

    # Process only common filenames
    for filename in tqdm(common_filenames, desc="Processing files"):
        # Get matching rows for this filename
        rows1 = df1_grouped.get_group(filename)
        rows2 = df2_grouped.get_group(filename)

        # Ensure both groups have the same number of rows
        min_rows = min(len(rows1), len(rows2))
        if min_rows == 0:
            continue

        # Calculate IoU for aligned rows and save directly to the DataFrame
        for idx in range(min_rows):
            row1 = rows1.iloc[idx]
            row2 = rows2.iloc[idx]
            iou = intersection_area(row1, row2) / union_area(row1, row2)
            df1.loc[row1.name, 'iou'] = iou
    return df1

# DO NOT RUN BELOW!, it takes about an hour of calculation, I already ran it on my end, the results are stored in file.json
def get_and_write_results():
    iou_results = calculate_iou_optimized('training.xlsx', 'predictions_training.xlsx')
    print(f"Calculated {len(iou_results)} IoU values")
    iou_results.to_excel("iou_results.xlsx") # it's just the train df with the IOU results in the columns

def mean_IOU(input):
    return np.average(input)

# Run this to get the data instead and continue using this
# df = pd.read_excel("iou_results.xlsx")