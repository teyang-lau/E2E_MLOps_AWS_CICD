from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import logging
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# Hyperparameters and algorithm parameters are described here
parser.add_argument("--num_round", type=int, default=100)  # CHANGE TO 100
parser.add_argument("--max_depth", type=int, default=3)
parser.add_argument("--eta", type=float, default=0.2)
parser.add_argument("--gamma", type=int, default=4)
parser.add_argument("--min_child_weight", type=int, default=6)
parser.add_argument("--subsample", type=float, default=0.9)
parser.add_argument("--objective", type=str, default="reg:squarederror")
parser.add_argument("--eval_metric", type=str, default="mae")
parser.add_argument("--nfold", type=int, default=2)  # 3
parser.add_argument("--early_stopping_rounds", type=int, default=3)

# Set location of input training data
parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
# Set location of input validation data
parser.add_argument(
    "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
)
# Set location where trained model will be stored. Default set by SageMaker, /opt/ml/model
parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
# Set location where model artifacts will be stored. Default set by SageMaker, /opt/ml/output/data
parser.add_argument(
    "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
)

args = parser.parse_args()
print(args)

logger.debug("Reading train data.")
data_train = pd.read_csv(f"{args.train}/train.csv")
train = data_train.drop("resale_price", axis=1)
label_train = pd.DataFrame(data_train["resale_price"])
dtrain = xgb.DMatrix(train, label=label_train)

logger.debug("Reading validation data.")
data_val = pd.read_csv(f"{args.validation}/validation.csv")
val = data_val.drop("resale_price", axis=1)
label_val = pd.DataFrame(data_val["resale_price"])
dval = xgb.DMatrix(val, label=label_val)

# Choose XGBoost model hyperparameters
params = {
    "max_depth": args.max_depth,
    "eta": args.eta,
    "gamma": args.gamma,
    "min_child_weight": args.min_child_weight,
    "subsample": args.subsample,
    "objective": args.objective,
}

num_boost_round = args.num_round
nfold = args.nfold
early_stopping_rounds = args.early_stopping_rounds

# Cross-validate train XGBoost model
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    nfold=nfold,
    early_stopping_rounds=early_stopping_rounds,
    metrics=[args.eval_metric],
    seed=2023,
)

logger.debug("Training xgboost model.")
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))

logger.debug("Generating predictions for train set")
train_pred = model.predict(dtrain)
logger.debug("Generating predictions for validation set")
val_pred = model.predict(dval)

logger.debug("Computing metrics for train set")
train_mae = mean_absolute_error(label_train, train_pred)
logger.debug("Computing metrics for validation set")
val_mae = mean_absolute_error(label_val, val_pred)

# print(f"train-mae: {train_mae:.2f}")
# print(f"validation-mae: {val_mae:.2f}")

metrics_data = {
    "hyperparameters": params,
    "reg_metrics": {
        "validation:auc": {"value": val_mae},
        "train:auc": {"value": train_mae},
    },
}

# Save the evaluation metrics to the location specified by output_data_dir
metrics_location = args.output_data_dir + "/metrics.json"
logger.info("Saving evaluation metrics to {}".format(metrics_location))
with open(metrics_location, "w") as f:
    json.dump(metrics_data, f)

# Save the trained model to the location specified by model_dir
model_location = args.model_dir + "/xgboost-model"
logger.info("Saving trained model to {}".format(model_location))
with open(model_location, "wb") as f:
    joblib.dump(model, f)
