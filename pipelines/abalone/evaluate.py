"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgb.DMatrix(df.values)

    logger.info("Generating predictions for test data.")
    predictions = model.predict(X_test)

    logger.debug("Calculating root mean squared error.")
    rmse = mean_squared_error(y_test, predictions, squared=False)
    std = np.std(y_test - predictions)
    metric_dict = {
        "regression_metrics": {
            "rmse": {"value": rmse, "standard_deviation": std},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with rmse: %f", rmse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(metric_dict))
