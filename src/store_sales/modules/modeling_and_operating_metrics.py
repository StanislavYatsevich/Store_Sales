import numpy as np
import pandas as pd
import xgboost as xgb
import json
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.base import RegressorMixin, clone
from typing import Tuple, Union
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from store_sales.modules import encode_features, EPSILON, N_SPLITS


def get_mae(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    model: RegressorMixin,
) -> float:
    """Returns MAE (Mean Absolute Error for a given model and given data).

    Returns MAE (Mean Absolute Error for a given model (Regressor) and
    given train and test parts of data.

    Args:
        X_train: pd.DataFrame instance representing the train part of the data
            without target variable values.
        X_test: pd.DataFrame instance representing the test part of the data
            without target variable values.
        y_train: pd.Series or np.ndarray with target variable values for X_train.
        y_test: pd.Series or np.ndarray with target variable values for X_test.
        model: RegressorMixin model used for fitting and a prediction.

    Returns:
        Mean Absolute Error between y_test and the model's prediction.
    """
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    mae_score = mean_absolute_error(y_test, y_pred)
    return mae_score


def get_models_and_metrics_cross_validation(
    train_data: pd.DataFrame, model: RegressorMixin
) -> Tuple[
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
]:
    """Calculates and saves metrics and models fitted using the cross-validation
    technique.

    Splits the train_data by all unique pairs (store_number, item_family).
    For each pair fits and saves the model and metrics calculated using the
    cross-validation technique with n_splits = 5.

    Args:
        train_data: pd.DataFrame instance representing the train part of the data.
        model: RegressorMixin model used for fitting and a prediction.

    Returns:
        A Tuple(dict, dict, dict, dict) where first three dictionaries represent
        the collection of metrics (MAE, Average Sales, WMAPE) for each pair
        (store_number, item_family) and the 4th one represents the collection of
        fitted models.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    mae_scores = dict()
    avg_sales = dict()
    wmape_percentage_scores = dict()
    models = dict()

    for store_num in train_data["store_number"].unique():
        for item_family in train_data["item_family"].unique():
            mae_scores_this_split = []
            data = train_data[
                (train_data["store_number"] == store_num)
                & (train_data["item_family"] == item_family)
            ]
            X = data.drop(["item_sales"], axis=1)
            y = data["item_sales"]
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                X_train_encoded, X_test_encoded = encode_features(X_train, X_test)
                model_clone = clone(model)
                mae = get_mae(
                    X_train_encoded, X_test_encoded, y_train, y_test, model_clone
                )
                mae_scores_this_split.append(mae)

            avg_sales_this_series = np.round(np.mean(data["item_sales"]), 2)
            mae_this_series = np.round(np.mean(np.array(mae_scores_this_split)), 2)
            wmape_percentage_this_series = np.round(
                100 * mae_this_series / (avg_sales_this_series + EPSILON), 2
            )

            avg_sales[(store_num, item_family)] = avg_sales_this_series
            mae_scores[(store_num, item_family)] = mae_this_series
            wmape_percentage_scores[(store_num, item_family)] = (
                wmape_percentage_this_series
            )
            models[(store_num, item_family)] = model_clone

    return mae_scores, avg_sales, wmape_percentage_scores, models


def save_metrics_and_models(
    metrics_file: Union[str, Path],
    models_file: Union[str, Path],
    mae_scores: dict[Tuple[int, str], float],
    avg_sales: dict[Tuple[int, str], float],
    wmape_percentage_scores: dict[Tuple[int, str], float],
    models: dict[Tuple[int, str], xgb.XGBRegressor],
) -> None:
    """Saves metrics and models locally.

    Saves given metrics and fitted models from get_models_and_metrics_cross_validation()
    function locally to given files of .json and .pkl extensions respectively.
    Prints the report of a successful saving.

    Args:
        metrics_file: str or Path instance of a file path where metrics are
            saved to.
        models_file: str or Path instance of a file path where models are
            saved to.
        mae_scores: a dictionary of MAE scores for every pair (store_number,
            item_family).
        avg_sales: a dictionary of Average sales for every pair (store_number,
            item_family).
        wmape_percentage_scores: a dictionary of WMAPE scores for every pair
            (store_number, item_family).
        models: a dictionary of models for every pair (store_number, item_family).
    """
    metrics = {
        "MAE scores": {str(k): v for k, v in mae_scores.items()},
        "Mean sales": {str(k): v for k, v in avg_sales.items()},
        "WMAPE scores, %": {str(k): v for k, v in wmape_percentage_scores.items()},
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics are successfully saved to {metrics_file}")

    with open(models_file, "wb") as f:
        pickle.dump(models, f)
    print(f"Models are successfully saved to {models_file}")
