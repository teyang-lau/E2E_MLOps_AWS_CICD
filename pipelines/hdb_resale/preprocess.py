"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

LABEL_COLUMN = "resale_price"

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    logger.info("Received arguments {}".format(args))

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/resale-flat-prices.csv"  # download data as this name
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    data = pd.read_csv(fn)
    os.unlink(fn)

    data = data.sort_values("month").reset_index(drop=True)
    columns = [
        "town",
        "flat_type",
        "storey_range",
        "floor_area_sqm",
        "flat_model",
        "lease_commence_date",
        "remaining_lease",
        "resale_price",
    ]
    data = data[columns]

    logger.debug("Replacing uncommon flat names from data.")
    data = data.replace(regex=[r".*[mM]aisonette.*", "foo"], value="Maisonette")

    logger.debug("Extracting number from remaining_lease.")
    data["remaining_lease"] = data["remaining_lease"].str.extract(r"(\d+)(?= years)")
    data = data.astype({"remaining_lease": "int16"})

    logger.debug("Label encoding categorical columns.")
    cat = data["storey_range"].astype("category")
    data["storey_range"] = cat.cat.codes
    storey_range_map = dict(enumerate(cat.cat.categories))

    flat_type_map = {
        "1 ROOM": 0,
        "2 ROOM": 1,
        "3 ROOM": 2,
        "4 ROOM": 3,
        "5 ROOM": 4,
        "MULTI-GENERATION": 5,
        "EXECUTIVE": 6,
    }
    data = data.replace({"flat_type": flat_type_map})

    logger.debug("One-hot encoding categorical columns.")
    data = pd.get_dummies(
        data,
        columns=["town"],
        prefix=["town"],
        dtype=int,
        drop_first=True,
    )  # central is baseline
    data = pd.get_dummies(data, columns=["flat_model"], prefix=["model"], dtype="int8")
    # remove standard, setting it as the baseline
    data = data.drop("model_Standard", axis=1)

    data_processed = data.copy()

    # Split into train, val, test
    y = data_processed[LABEL_COLUMN]
    X = data_processed.drop([LABEL_COLUMN], axis=1)

    TRAIN_RATIO = args.train_ratio
    VAL_RATIO = args.val_ratio
    TEST_RATIO = args.test_ratio

    logger.debug("Splitting data into train, validation, and test sets")
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X,
        y,
        test_size=1 - TRAIN_RATIO,
        random_state=2023,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test,
        y_val_test,
        test_size=(TEST_RATIO / (TEST_RATIO + VAL_RATIO)),
        random_state=2023,
    )

    # set y as first column
    train_df = pd.concat([y_train, X_train], axis=1)
    val_df = pd.concat([y_val, X_val], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)
    dataset_df = pd.concat([y, X], axis=1)

    logger.info("Train data shape after preprocessing: {}".format(train_df.shape))
    logger.info("Validation data shape after preprocessing: {}".format(val_df.shape))
    logger.info("Test data shape after preprocessing: {}".format(test_df.shape))

    # logger.debug("Defining transformers.")
    # numeric_features = list(feature_columns_names)
    # numeric_features.remove("sex")
    # numeric_transformer = Pipeline(
    #     steps=[
    #         ("imputer", SimpleImputer(strategy="median")),
    #         ("scaler", StandardScaler()),
    #     ]
    # )

    # categorical_features = ["sex"]
    # categorical_transformer = Pipeline(
    #     steps=[
    #         ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    #         ("onehot", OneHotEncoder(handle_unknown="ignore")),
    #     ]
    # )

    # preprocess = ColumnTransformer(
    #     transformers=[
    #         ("num", numeric_transformer, numeric_features),
    #         ("cat", categorical_transformer, categorical_features),
    #     ]
    # )

    # logger.info("Applying transforms.")
    # y = df.pop("rings")
    # X_pre = preprocess.fit_transform(df)
    # y_pre = y.to_numpy().reshape(len(y), 1)

    # X = np.concatenate((y_pre, X_pre), axis=1)

    # logger.info(
    #     "Splitting %d rows of data into train, validation, test datasets.", len(X)
    # )
    # np.random.shuffle(X)
    # train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    # Save processed datasets to the local paths
    train_output_path = os.path.join(f"{base_dir}/train", "train.csv")
    val_output_path = os.path.join(f"{base_dir}/validation", "validation.csv")
    test_output_path = os.path.join(f"{base_dir}/test", "test.csv")

    logger.info("Saving train data to {}".format(train_output_path))
    train_df.to_csv(train_output_path, index=False)
    logger.info("Saving validation data to {}".format(val_output_path))
    val_df.to_csv(val_output_path, index=False)
    logger.info("Saving test data to {}".format(test_output_path))
    test_df.to_csv(test_output_path, index=False)

    # logger.info("Writing out datasets to %s.", base_dir)
    # pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    # pd.DataFrame(validation).to_csv(
    #     f"{base_dir}/validation/validation.csv", header=False, index=False
    # )
    # pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
