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


def get_models_and_metrics_cv_and_testing(
    train_data: pd.DataFrame, test_data: pd.DataFrame, model: RegressorMixin
) -> Tuple[
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
    dict[Tuple[int, str], float],
]:
    """Calculates and saves metrics calculated during cross-validation and 
    both metrics and fitted models during the final model evaluation.

    Splits the data by all unique pairs (store_number, item_family).
    For each pair fits and saves the model and metrics.

    Args:
        train_data: pd.DataFrame instance representing the train part of the data.
        test_data: pd.DataFrame instance representing the test part of the data.
        model: RegressorMixin model used for fitting and a prediction.

    Returns:
        Tuple(dict, dict, dict, dict, dict, dict, dict) where first three dictionaries represent
        the collection of cross-validation metrics (MAE, Average Sales, WMAPE) for each pair
        (store_number, item_family), second three dictionaries - the same collection of testing
        metrics and the last dictionary - the collection of models.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    mae_scores_cv = dict()
    avg_sales_cv = dict()
    wmape_percentage_scores_cv = dict()

    mae_scores_test = dict()
    avg_sales_test = dict()
    wmape_percentage_scores_test = dict()

    models_fitted = dict()

    for store_num in train_data["store_number"].unique():
        for item_family in train_data["item_family"].unique():
            mae_scores_this_split = []
            train_data = train_data[
                (train_data["store_number"] == store_num)
                & (train_data["item_family"] == item_family)
            ]
            X = train_data.drop(["item_sales"], axis=1)
            y = train_data["item_sales"]
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                X_train_encoded, X_test_encoded = encode_features(X_train, X_test)
                model_clone_cv = clone(model)
                mae_cv = get_mae(
                    X_train_encoded, X_test_encoded, y_train, y_test, model_clone_cv
                )
                mae_scores_this_split.append(mae_cv)

            avg_sales_this_series_cv = np.round(np.mean(train_data["item_sales"]), 2)
            mae_this_series_cv = np.round(np.mean(np.array(mae_scores_this_split)), 2)
            wmape_percentage_this_series_cv = np.round(
                100 * mae_this_series_cv / (avg_sales_this_series_cv + EPSILON), 2
            )

            avg_sales_cv[(store_num, item_family)] = avg_sales_this_series_cv
            mae_scores_cv[(store_num, item_family)] = mae_this_series_cv
            wmape_percentage_scores_cv[(store_num, item_family)] = (
                wmape_percentage_this_series_cv
            )

            test_data = test_data[
                (test_data["store_number"] == store_num)
                & (test_data["item_family"] == item_family)
            ]
            X_train = train_data.drop(["item_sales"], axis=1)
            y_train = train_data["item_sales"]
            X_test = test_data.drop(["item_sales"], axis=1)
            y_test = test_data["item_sales"]
            X_train_encoded, X_test_encoded = encode_features(X_train, X_test)
            model_clone_test = clone(model)

            model_clone_test.fit(X_train_encoded, y_train)
            y_pred = pd.Series(
                model_clone_test.predict(X_test_encoded), index=y_test.index
            )

            mae_this_series_test = np.round(mean_absolute_error(y_test, y_pred), 2)
            avg_sales_this_series_test = np.round(np.mean(test_data["item_sales"]), 2)
            wmape_percentage_this_series_test = np.round(
                100 * mae_this_series_test / (avg_sales_this_series_test + EPSILON), 2
            )

            mae_scores_test[(store_num, item_family)] = mae_this_series_test
            avg_sales_test[(store_num, item_family)] = avg_sales_this_series_test
            wmape_percentage_scores_test[(store_num, item_family)] = (
                wmape_percentage_this_series_test
            )

            models_fitted[(store_num, item_family)] = model_clone_test

    return (
        mae_scores_cv,
        avg_sales_cv,
        wmape_percentage_scores_cv,
        mae_scores_test,
        avg_sales_test,
        wmape_percentage_scores_test,
        models_fitted,
    )


def save_metrics_and_models(
    metrics_cv_file: Union[str, Path],
    metrics_test_file: Union[str, Path],
    models_file: Union[str, Path],
    mae_scores_cv: dict[Tuple[int, str], float],
    avg_sales_cv: dict[Tuple[int, str], float],
    wmape_percentage_scores_cv: dict[Tuple[int, str], float],
    mae_scores_test: dict[Tuple[int, str], float],
    avg_sales_test: dict[Tuple[int, str], float],
    wmape_percentage_scores_test: dict[Tuple[int, str], float],
    models: dict[Tuple[int, str], xgb.XGBRegressor],
) -> None:
    """Saves metrics and models locally.

    Saves given metrics and fitted models from get_models_and_metrics_cv_and_testing()
    function locally to given files of .json and .pkl extensions respectively.
    Prints the reports of a successful saving.

    Args:
        metrics_cv_file: str or Path instance of a file path where cross-validation metrics
            are saved to.
        metrics_test_file: str or Path instance of a file path where testing metrics are
            saved to.
        models_file: str or Path instance of a file path where models are
            saved to.
        mae_scores_cv: a dictionary of cross-validation MAE scores for every pair
            (store_number, item_family).
        avg_sales_cv: a dictionary of cross-validation Average sales for every pair
            (store_number, item_family).
        wmape_percentage_scores_cv: a dictionary of cross-validation WMAPE scores for 
            every pair (store_number, item_family).
        mae_scores_test: a dictionary of testing MAE scores for every pair
            (store_number, item_family).
        avg_sales_test: a dictionary of testing Average sales for every pair
            (store_number, item_family).
        wmape_percentage_scores_test: a dictionary of testing WMAPE scores for 
            every pair (store_number, item_family).
        models: a dictionary of models for every pair (store_number, item_family).
    """
    metrics_cv = {
        "MAE scores": {str(k): v for k, v in mae_scores_cv.items()},
        "Mean sales": {str(k): v for k, v in avg_sales_cv.items()},
        "WMAPE scores, %": {str(k): v for k, v in wmape_percentage_scores_cv.items()},
    }
    metrics_test = {
        "MAE scores": {str(k): v for k, v in mae_scores_test.items()},
        "Mean sales": {str(k): v for k, v in avg_sales_test.items()},
        "WMAPE scores, %": {str(k): v for k, v in wmape_percentage_scores_test.items()},
    }

    with open(metrics_cv_file, "w") as f:
        json.dump(metrics_cv, f, indent=4)
    print(f"Cross-validation metrics are successfully saved to {metrics_cv_file}")

    with open(metrics_test_file, "w") as f:
        json.dump(metrics_test, f, indent=4)
    print(f"Testing metrics are successfully saved to {metrics_test_file}")

    with open(models_file, "wb") as f:
        pickle.dump(models, f)
    print(f"Models are successfully saved to {models_file}")
