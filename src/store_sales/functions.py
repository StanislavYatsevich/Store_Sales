import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.base import RegressorMixin, clone
from typing import List, Any, Tuple, Union
from sklearn.model_selection import TimeSeriesSplit
from path import Path
import optuna
import xgboost as xgb
import mlflow
import mlflow.sklearn
import json
import pickle


def prepare_data(
    data: pd.DataFrame,
    holidays_events_data: pd.DataFrame,
    oil_data: pd.DataFrame,
    stores_data: pd.DataFrame,
) -> pd.DataFrame:

    holidays_events_data["priority"] = holidays_events_data["locale"].map(
        {"National": 3, "Regional": 2, "Local": 1}
    )
    holidays_events_data = holidays_events_data.sort_values(
        by=["date", "priority"], ascending=False
    )
    holidays_events_data = holidays_events_data.drop_duplicates(
        subset=["date"], keep="first"
    )
    holidays_events_data.drop("priority", axis=1, inplace=True)

    data = pd.merge(data, stores_data, on=["store_nbr"], how="inner")
    data = pd.merge(data, oil_data, on=["date"], how="left")
    data = pd.merge(data, holidays_events_data, on=["date"], how="left")
    data.fillna(
        {
            "type_y": "Not holiday",
            "locale": "Not holiday",
            "locale_name": "Not holiday",
            "description": "Not holiday",
            "transferred": "Not holiday",
        },
        inplace=True,
    )
    data.rename(
        columns={
            "store_nbr": "store_number",
            "type_x": "store_type",
            "cluster": "store_cluster",
            "dcoilwtico": "oil_price",
            "locale": "holiday_status",
            "locale_name": "holiday_location",
            "description": "holiday_description",
            "type_y": "day_type",
            "transferred": "is_holiday_transferred",
            "family": "item_family",
            "sales": "item_sales",
            "onpromotion": "items_on_promotion",
        },
        inplace=True,
    )
    data["date"] = pd.to_datetime(data["date"])
    data.set_index("id", inplace=True)
    data["oil_price"].bfill(inplace=True)
    data["is_holiday_transferred"] = data["is_holiday_transferred"].map(
        lambda x: False if not x or x == "Not holiday" else True
    )

    data = data.sort_values(by=["store_number", "item_family", "date"])
    data["mean_sales_prev_month"] = data.groupby(["store_number", "item_family"])[
        "item_sales"
    ].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    data["mean_sales_prev_month"] = data["mean_sales_prev_month"].fillna(method="bfill")
    data = data.sort_values(by=["date", "store_number", "item_family"])
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    def is_during_falling_period(date, periods):
        for start, end in periods:
            if start <= date <= end:
                return 1
        return 0

    oil_price_falling_start_1 = pd.to_datetime("2014-07-01")
    oil_price_falling_finish_1 = pd.to_datetime("2015-01-31")
    oil_price_falling_start_2 = pd.to_datetime("2015-06-01")
    oil_price_falling_finish_2 = pd.to_datetime("2016-02-29")
    data["date"] = pd.to_datetime(data["date"])
    periods = [
        (oil_price_falling_start_1, oil_price_falling_finish_1),
        (oil_price_falling_start_2, oil_price_falling_finish_2),
    ]
    data["is_during_oil_prices_falling"] = data["date"].apply(
        lambda x: is_during_falling_period(x, periods)
    )

    def is_special_unit(unit: Any, special_list: List[Any]) -> int:
        if unit in special_list:
            return 1
        return 0

    popular_stores = [3, 8, 11, 44, 45, 46, 47, 48, 49, 50, 51]
    popular_clusters = [5, 8, 11, 14, 17]
    non_popular_clusters = [7]
    special_non_working_days = ["Additional", "Bridge", "Transfer", "Event"]
    popular_holidays = ["National"]
    popular_states = ["Pichincha"]
    non_popular_states = ["Manabi", "Pastaza"]
    popular_cities = ["Quito", "Cayambe"]
    non_popular_cities = ["Manta", "Puyo"]
    popular_store_types = ["A"]
    date_of_earthquake = pd.to_datetime("2016-04-16")

    data["is_popular_store"] = data["store_number"].apply(
        lambda x: is_special_unit(x, popular_stores)
    )
    data["is_popular_cluster"] = data["store_cluster"].apply(
        lambda x: is_special_unit(x, popular_clusters)
    )
    data["is_non_popular_cluster"] = data["store_cluster"].apply(
        lambda x: is_special_unit(x, non_popular_clusters)
    )
    data["is_special_non_working_day"] = data["day_type"].apply(
        lambda x: is_special_unit(x, special_non_working_days)
    )
    data["is_national_holiday"] = data["holiday_status"].apply(
        lambda x: is_special_unit(x, popular_holidays)
    )
    data["is_state_pichincha"] = data["state"].apply(
        lambda x: is_special_unit(x, popular_states)
    )
    data["is_state_manabi_or_pastaza"] = data["state"].apply(
        lambda x: is_special_unit(x, non_popular_states)
    )
    data["is_city_quito_or_cayambe"] = data["city"].apply(
        lambda x: is_special_unit(x, popular_cities)
    )
    data["is_city_manta_or_puyo"] = data["city"].apply(
        lambda x: is_special_unit(x, non_popular_cities)
    )
    data["is_store_type_A"] = data["store_type"].apply(
        lambda x: is_special_unit(x, popular_store_types)
    )
    data["number_of_days_since_earthquake"] = (
        data["date"] - date_of_earthquake
    ).dt.days
    return data


def encode_features(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    min_date = pd.to_datetime(train_data["date"]).min()

    def add_date_features(data: pd.DataFrame) -> pd.DataFrame:
        data["date"] = pd.to_datetime(data["date"])
        data["days_since_start"] = (pd.to_datetime(data["date"]) - min_date).dt.days
        data["year"] = pd.to_datetime(data["date"]).dt.year
        data["month"] = pd.to_datetime(data["date"]).dt.month
        data["day_of_week"] = pd.to_datetime(data["date"]).dt.dayofweek
        data.drop(["date"], axis=1, inplace=True)
        return data

    train_data = add_date_features(train_data)
    test_data = add_date_features(test_data)

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

    cat_columns = [
        "item_family",
        "city",
        "state",
        "store_type",
        "day_type",
        "holiday_status",
        "holiday_location",
        "holiday_description",
        "is_holiday_transferred",
    ]

    train_data[cat_columns] = ordinal_encoder.fit_transform(train_data[cat_columns])
    test_data[cat_columns] = ordinal_encoder.transform(test_data[cat_columns])

    return train_data, test_data


def get_tree_based_predicts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    models_list: List[RegressorMixin],
    models_names: List[str],
) -> Tuple[pd.DataFrame, List[float], List[float]]:
    df = pd.concat([y_train, y_test])
    mae_scores = []
    mape_scores = []

    for model in models_list:
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test), index=y_test.index)
        predict = pd.concat([y_train, y_pred])
        df = pd.concat([df, predict], axis=1)
        mae_scores.append(np.round(mean_absolute_error(y_test, y_pred), 2))
        mape_scores.append(
            100 * np.round(mean_absolute_percentage_error(y_test, y_pred), 2)
        )

    df.columns = ["Real Sales"] + models_names
    df = pd.concat([pd.concat([X_train, X_test], axis=0), df], axis=1)
    return (df, mae_scores, mape_scores)


def get_mae(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    model: RegressorMixin,
) -> float:
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    mae_score = mean_absolute_error(y_test, y_pred)
    return mae_score


def get_models_and_metrics_cross_validation(
    train_data: pd.DataFrame, model: RegressorMixin
) -> Tuple[float, float, float]:
    tscv = TimeSeriesSplit(n_splits=5)
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
                X_train_encoded, X_test_encoded = encode_features(
                    X_train.copy(), X_test.copy()
                )
                model_clone = clone(model)
                model_clone.fit(X_train_encoded, y_train)
                y_pred = pd.Series(
                    model_clone.predict(X_test_encoded), index=y_test.index
                )
                mae = mean_absolute_error(y_test, y_pred)
                mae_scores_this_split.append(mae)

            epsilon = 10 ** (-5)
            avg_sales_this_series = np.round(np.mean(data["item_sales"]), 2)
            mae_this_series = np.round(np.mean(np.array(mae_scores_this_split)), 2)
            wmape_percentage_this_series = np.round(
                100 * mae_this_series / (avg_sales_this_series + epsilon), 2
            )

            avg_sales[(store_num, item_family)] = avg_sales_this_series
            mae_scores[(store_num, item_family)] = mae_this_series
            wmape_percentage_scores[(store_num, item_family)] = (
                wmape_percentage_this_series
            )
            models[(store_num, item_family)] = model_clone

    return mae_scores, avg_sales, wmape_percentage_scores, models


def optimize_xgboost_params_with_optuna(
    train_data: pd.DataFrame, shops_number: int, n_trials: int
) -> Tuple[xgb.XGBRegressor, dict]:
    tscv = TimeSeriesSplit(n_splits=5)
    random_stores = np.random.choice(
        train_data["store_number"].unique(), shops_number, replace=False
    )

    def objective(trial) -> float:
        parameters = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }
        model = xgb.XGBRegressor(**parameters, random_state=42, n_jobs=-1)
        mae_scores = []
        for store_num in random_stores:
            for item_family in train_data["item_family"].unique():
                data = train_data[
                    (train_data["store_number"] == store_num)
                    & (train_data["item_family"] == item_family)
                ]
                X = data.drop(["item_sales"], axis=1)
                y = data["item_sales"]
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    X_train_encoded, X_test_encoded = encode_features(
                        X_train.copy(), X_test.copy()
                    )
                    mae = get_mae(
                        X_train_encoded, X_test_encoded, y_train, y_test, model
                    )
                    mae_scores.append(mae)

        return np.array(mae_scores).mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)

    return best_model, best_params


def init_mlflow_experiment(tracking_server_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_server_uri)
    mlflow.set_experiment(experiment_name)


def save_metrics_and_models(
    metrics_file: Union[str, Path],
    models_file: Union[str, Path],
    mae_scores: dict[Tuple[int, str], float],
    avg_sales: dict[Tuple[int, str], float],
    wmape_percentage_scores: dict[Tuple[int, str], float],
    models: dict[Tuple[int, str], xgb.XGBRegressor],
) -> None:
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
