import psutil
import os
import resource
import pandas as pd
import numpy as np

_call_counter = {"n": 0}

cost = {
    # Scania Component X official cost matrix for punishing missclasifications.
    # Source: ../data/2024-34-1/documentation/2024_IDA_challenge_v2.pdf
    (0, 1): 7,
    (0, 2): 8,
    (0, 3): 9,
    (0, 4): 10,
    (1, 0): 200,
    (1, 2): 7,
    (1, 3): 8,
    (1, 4): 9,
    (2, 0): 300,
    (2, 1): 200,
    (2, 3): 7,
    (2, 4): 8,
    (3, 0): 400,
    (3, 1): 300,
    (3, 2): 200,
    (3, 4): 7,
    (4, 0): 500,
    (4, 1): 400,
    (4, 2): 300,
    (4, 3): 200,
}

# matrix form of the same costs, _C[true, pred], diagonal = 0
_C = np.array([[cost.get((i, j), 0) for j in range(5)] for i in range(5)])

# class mix of each vehicle's final label, set once with set_prior()
_prior = None


def set_prior(y_train):
    """Call once before scoring. y_train must be indexed by vehicle id."""
    global _prior
    _prior = (
        y_train.groupby(y_train.index)
        .last()
        .value_counts(normalize=True)
        .reindex(range(5), fill_value=0)
        .values
    )
    print("prior:", np.round(_prior, 4))


def per_vehicle_cost_score(estimator, X, y):
    if _prior is None:
        raise RuntimeError(
            "Prior not set. Call set_prior(y_train) once before scoring "
            "(training labels, indexed by vehicle id)."
        )

    _call_counter["n"] += 1
    fold_num = _call_counter["n"]

    # undo the balanced weighting, then pick the lowest expected-cost class
    proba = estimator.predict_proba(X) * _prior
    proba /= proba.sum(axis=1, keepdims=True)
    y_pred = estimator.classes_[(proba @ _C).argmin(axis=1)]

    df = pd.DataFrame({"y_true": np.asarray(y), "y_pred": y_pred}, index=X.index)
    per_vehicle = df.groupby(df.index).last()
    total = sum(
        cost.get((t, p), 0)
        for t, p in zip(per_vehicle["y_true"], per_vehicle["y_pred"])
    )

    current_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB
    print(
        f"[fold {fold_num}] score={-total}  current_RSS={current_gb:.2f}GB  peak_RSS_so_far={peak_gb:.2f}GB"
    )

    return -total
