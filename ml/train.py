import joblib
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.pipeline import Pipeline as ImbPipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import GroupKFold
from metrics import per_vehicle_cost_score

np.random.seed(1)

train_data = pd.read_parquet("../data/fe_data/temporal_train.parquet")
print(train_data.memory_usage(deep=True).sum() / 1e9, "GB (deep)")
train = train_data.set_index("vehicle_id")

vehicle_ids = train.index.unique()

train_vids, val_vids = train_test_split(vehicle_ids, test_size=0.2, random_state=8)

X_train = train.loc[train.index.isin(train_vids)].drop(columns="class_label")
y_train = train.loc[train.index.isin(train_vids), "class_label"]
# X_val = train.loc[train.index.isin(val_vids)].drop(columns="class_label")
# y_val = train.loc[train.index.isin(val_vids), "class_label"]

print("train vars set")

estimators = [
    ("encoder", TargetEncoder()),
    ("clf", XGBClassifier(random_state=8, n_jobs=1, tree_method="hist")),
]

pipe = ImbPipeline(steps=estimators)
pipe.set_output(transform="pandas")


search_space = {
    "clf__max_depth": Integer(4, 6),
    "clf__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
    "clf__subsample": Real(0.5, 1.0),
    "clf__reg_lambda": Real(0.0, 10.0),
}

groups = X_train.index

model = BayesSearchCV(
    pipe,
    search_space,
    cv=GroupKFold(n_splits=3),
    n_iter=25,
    scoring=per_vehicle_cost_score,
    random_state=8,
    n_jobs=1,
    verbose=3,
)

sample_weight = compute_sample_weight("balanced", y_train)

model.fit(
    X_train,
    y_train,
    # eval_set=[(X_val, y_val)],
    groups=X_train.index,
    clf__sample_weight=sample_weight,
    clf__verbose=True,
)

joblib.dump(model, "../ml/model/FleetBoost")

print("done")
