"""
Scores the trained fleetnet model against the official SCANIA validation
set, restricted to only the rows that carry a real label (each vehicle's
single truncated last readout, matching validation_labels.csv).

This is the one number that's actually comparable to the published
benchmarks (baseline ~57,400; best published result ~36,113) -- unlike
the internal train/test CV scores, which are drawn from the training
population and use a different, harder true-class distribution (see
prior discussion: training vehicles' natural last readout sits very
close to failure, while official validation's truncation point is
genuinely random).
"""

from math import log
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import log_loss
from metrics import per_vehicle_cost_score, set_prior

# --- Load the trained model ---
model = joblib.load("../ml/model/FleetBoost")

print("models loaded")

# --- Load the official validation set (already feature-engineered) ---
val_data = pd.read_parquet("../data/fe_data/temporal_val.parquet")
val = val_data.set_index("vehicle_id")  # keep vehicle_id as index, not a column

# --- Load the official test set (already feature-engineered) ---
test_data = pd.read_parquet("../data/fe_data/temporal_test.parquet")
test = test_data.set_index("vehicle_id")  # keep vehicle_id as index, not a column

# --- Keep only the rows that actually carry a real label ---
# Every vehicle has many rows in this file, but only its last (truncated)
# readout has a non-NaN class_label -- dropping NaN here naturally leaves
# exactly one row per vehicle, matching validation_labels.csv's structure.
scoreable_val = val.dropna(subset=["class_label"])
print(f"scoreable rows: {len(scoreable_val)} (expect 5,046 vehicles)")

scoreable_test = test.dropna(subset=["class_label"])
print(f"scoreable rows: {len(scoreable_test)} (expect 5,045 vehicles)")

X_val = scoreable_val.drop(columns=["class_label"])
y_val = scoreable_val["class_label"]

X_test = scoreable_test.drop(columns=["class_label"])
y_test = scoreable_test["class_label"]
# --- Score ---
# per_vehicle_cost_score's internal groupby(index).last() is a no-op here
# since scoreable_val already has exactly one row per vehicle -- kept for
# consistency with how the function is used elsewhere in the project.
set_prior(y_val)  # training labels, indexed by vehicle id
val_cost_score = per_vehicle_cost_score(model, X_val, y_val)

set_prior(y_test)  # training labels, indexed by vehicle id
test_cost_score = per_vehicle_cost_score(model, X_test, y_test)

print("VALIDATION COST SCORE:", val_cost_score)
print("Test COST SCORE:", test_cost_score)
print("(baseline is -57,400; published best result is -36,113)")
