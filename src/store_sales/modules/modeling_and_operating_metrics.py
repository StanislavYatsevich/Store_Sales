import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.base import RegressorMixin, clone
from typing import Tuple, Union
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from store_sales.modules import EPSILON, N_SPLITS


def calculate_daily_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Calculate MAE and WMAPE for each day."""
    daily_errors = pd.DataFrame(
        {
            "true_sales": y_true,
            "pred_sales": y_pred,
            "MAE": np.abs(y_true - y_pred),
            "WMAPE, %": 100 * np.abs(y_true - y_pred) / (y_true + EPSILON),
        }
    )
    daily_errors["day"] = daily_errors.index
    return np.round(
        daily_errors.groupby("day").agg({"MAE": "mean", "WMAPE, %": "mean"}), 2
    )


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
    dict[Tuple[int, str], float],
]:
    """Calculates and saves metrics and fitted models.

    Splits the data by all unique pairs (store_number, item_family). For each pair performs
    cross-validation, calculates and saves its metrics. After that fits the model on the
    whole train data, calculates metrics on the test data and saves both the model and the metrics.

    Args:
        train_data: pd.DataFrame instance representing the train part of the data.
        test_data: pd.DataFrame instance representing the test part of the data.
        model: RegressorMixin model.

    Returns:
        Tuple(dict, dict, dict, dict, dict, dict, dict, dict) where first three dictionaries represent
        the collection of cross-validation metrics (MAE, Average Sales, WMAPE) for each pair
        (store_number, item_family), then four dictionaries - the same collection of testing
        metrics + daily metrics and the last dictionary - the collection of models.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    mae_scores_cv = dict()
    avg_sales_cv = dict()
    wmape_percentage_scores_cv = dict()

    mae_scores_test = dict()
    avg_sales_test = dict()
    wmape_percentage_scores_test = dict()
    daily_metrics_test = dict()

    models_fitted = dict()

    cat_columns = [
        col for col in train_data.columns if train_data[col].dtype == "object"
    ]
    for col in cat_columns:
        train_data[col] = train_data[col].astype("category")
        test_data[col] = test_data[col].astype("category")

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
                model_clone_cv = clone(model)
                model_clone_cv.fit(X_train, y_train, categorical_feature=cat_columns)
                y_pred = pd.Series(model_clone_cv.predict(X_test), index=y_test.index)
                mae_cv = mean_absolute_error(y_test, y_pred)
                mae_scores_this_split.append(mae_cv)

            avg_sales_this_series_cv = np.round(np.mean(data["item_sales"]), 2)
            mae_this_series_cv = np.round(np.mean(np.array(mae_scores_this_split)), 2)
            wmape_percentage_this_series_cv = np.round(
                100 * mae_this_series_cv / (avg_sales_this_series_cv + EPSILON), 2
            )

            avg_sales_cv[(store_num, item_family)] = avg_sales_this_series_cv
            mae_scores_cv[(store_num, item_family)] = mae_this_series_cv
            wmape_percentage_scores_cv[(store_num, item_family)] = (
                wmape_percentage_this_series_cv
            )

            data = test_data[
                (test_data["store_number"] == store_num)
                & (test_data["item_family"] == item_family)
            ]
            X_train = X
            y_train = y
            X_test = data.drop(["item_sales"], axis=1)
            y_test = data["item_sales"]
            model_clone_test = clone(model)

            model_clone_test.fit(X_train, y_train, categorical_feature=cat_columns)
            y_pred = pd.Series(model_clone_test.predict(X_test), index=y_test.index)

            mae_this_series_test = np.round(mean_absolute_error(y_test, y_pred), 2)
            avg_sales_this_series_test = np.round(np.mean(data["item_sales"]), 2)
            wmape_percentage_this_series_test = np.round(
                100 * mae_this_series_test / (avg_sales_this_series_test + EPSILON), 2
            )

            mae_scores_test[(store_num, item_family)] = mae_this_series_test
            avg_sales_test[(store_num, item_family)] = avg_sales_this_series_test
            wmape_percentage_scores_test[(store_num, item_family)] = (
                wmape_percentage_this_series_test
            )

            daily_metrics_test[(store_num, item_family)] = calculate_daily_metrics(
                y_test, y_pred
            )

            models_fitted[(store_num, item_family)] = model_clone_test

    return (
        mae_scores_cv,
        avg_sales_cv,
        wmape_percentage_scores_cv,
        mae_scores_test,
        avg_sales_test,
        wmape_percentage_scores_test,
        daily_metrics_test,
        models_fitted,
    )


def save_metrics_and_models(
    metrics_cv_file: Union[str, Path],
    metrics_test_file: Union[str, Path],
    daily_metrics_file: Union[str, Path],
    models_file: Union[str, Path],
    mae_scores_cv: dict[Tuple[int, str], float],
    avg_sales_cv: dict[Tuple[int, str], float],
    wmape_percentage_scores_cv: dict[Tuple[int, str], float],
    mae_scores_test: dict[Tuple[int, str], float],
    avg_sales_test: dict[Tuple[int, str], float],
    wmape_percentage_scores_test: dict[Tuple[int, str], float],
    daily_metrics_test: dict[Tuple[int, str], pd.DataFrame],
    models: dict[Tuple[int, str], RegressorMixin],
) -> None:
    """Saves metrics and models locally.

    Saves given metrics and fitted models from get_models_and_metrics_cv_and_testing()
    function locally to given files of .json and .pkl extensions respectively.
    Prints the reports of a successful saving.

    Args:
        metrics_cv_file: file path where cross-validation metrics
            are saved to.
        metrics_test_file: file path where testing metrics are
            saved to.
        models_file: file path where models are
            saved to.
        mae_scores_cv: dictionary with cross-validation MAE scores for every pair
            (store_number, item_family).
        avg_sales_cv: dictionary with cross-validation Average sales for every pair
            (store_number, item_family).
        wmape_percentage_scores_cv: dictionary with cross-validation WMAPE scores for
            every pair (store_number, item_family).
        mae_scores_test: dictionary with testing MAE scores for every pair
            (store_number, item_family).
        avg_sales_test: dictionary with testing Average sales for every pair
            (store_number, item_family).
        wmape_percentage_scores_test: dictionary with testing WMAPE scores for
            every pair (store_number, item_family).
        daily_metrics_test: dictionary with testing daily scores for
            every pair (store_number, item_family).
        models: dictionary with fitted models for every pair (store_number, item_family).
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

    daily_metrics_test_serializable = {
        (store_num, item_family): df.to_dict()
        for (store_num, item_family), df in daily_metrics_test.items()
    }

    daily_metrics = {
        "Daily metrics": {str(k): v for k, v in daily_metrics_test_serializable.items()}
    }

    with open(metrics_cv_file, "w") as f:
        json.dump(metrics_cv, f, indent=4)
    print(f"Cross-validation metrics are successfully saved to {metrics_cv_file}")

    with open(metrics_test_file, "w") as f:
        json.dump(metrics_test, f, indent=4)
    print(f"Testing metrics are successfully saved to {metrics_test_file}")

    with open(daily_metrics_file, "w") as f:
        json.dump(daily_metrics, f, indent=4)
    print(f"Daily metrics are successfully saved to {daily_metrics_file}")

    with open(models_file, "wb") as f:
        pickle.dump(models, f)
    print(f"Models are successfully saved to {models_file}")
