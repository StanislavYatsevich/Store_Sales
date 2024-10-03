import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import click
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from pathlib import Path

# import lightgbm as lgb
from store_sales.modules import (
    get_models_and_metrics_cv_and_testing,
    calculate_daily_metrics,
    save_metrics_and_models,
    optimize_lgb_params_with_optuna,
    PREPARED_FINAL_DATA_FOLDER_PATH,
    METRICS_FOLDER_PATH,
    MODELS_FOLDER_PATH,
    DEFAULT_SERVER_URI,
    DEFAULT_EXPERIMENT_NAME,
    EPSILON,
    N_SPLITS,
    N_SHOPS_OPTUNA,
    N_TRIALS_OPTUNA,
)


@click.command()
@click.option(
    "--metrics_folder_path",
    default=METRICS_FOLDER_PATH,
    type=click.Path(),
    help="Path to the folder where metrics are saved to",
)
@click.option(
    "--models_folder_path",
    default=MODELS_FOLDER_PATH,
    type=click.Path(),
    help="Path to the folder where models are saved to",
)
@click.option(
    "--send_to_server",
    default=True,
    type=bool,
    help="Flag to send metrics and models to MLFlow server or not",
)
def save_and_log_data(metrics_folder_path, models_folder_path, send_to_server):
    metrics_folder_path = Path(metrics_folder_path)
    models_folder_path = Path(models_folder_path)

    metrics_folder_path.mkdir(parents=True, exist_ok=True)
    models_folder_path.mkdir(parents=True, exist_ok=True)

    metrics_cv_file = Path(metrics_folder_path) / "metrics_cv.json"
    metrics_test_file = Path(metrics_folder_path) / "metrics_test.json"
    daily_metrics_file = Path(metrics_folder_path) / "daily_metrics_test.json"

    models_file = Path(models_folder_path) / "models.pkl"

    train_data = pd.read_csv(PREPARED_FINAL_DATA_FOLDER_PATH / "train_data.csv")
    test_data = pd.read_csv(PREPARED_FINAL_DATA_FOLDER_PATH / "test_data.csv")

    optimized_model, best_params = optimize_lgb_params_with_optuna(
        train_data, N_SHOPS_OPTUNA, N_TRIALS_OPTUNA
    )

    """params_lgb = {
        "max_depth": 2,
        "n_estimators": 115,
        "learning_rate": 0.02424128710744379,
        "subsample": 0.6976019464448409,
        "colsample_bytree": 0.9747490499129191,
        "lambda_l1": 1.2591101610869463e-06,
        "lambda_l2": 9.766607974404143e-05,
        "min_split_gain": 0.2651803122030023,
        "verbose": -1,
    }

    optimized_model = lgb.LGBMRegressor(**params_lgb, random_state=42, n_jobs=-1)"""

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    if not send_to_server:
        (
            mae_scores_cv,
            avg_sales_cv,
            wmape_percentage_scores_cv,
            mae_scores_test,
            avg_sales_test,
            wmape_percentage_scores_test,
            daily_metrics_test,
            models,
        ) = get_models_and_metrics_cv_and_testing(
            train_data, test_data, optimized_model
        )
        save_metrics_and_models(
            metrics_cv_file,
            metrics_test_file,
            daily_metrics_file,
            models_file,
            mae_scores_cv,
            avg_sales_cv,
            wmape_percentage_scores_cv,
            mae_scores_test,
            avg_sales_test,
            wmape_percentage_scores_test,
            daily_metrics_test,
            models,
        )
    else:
        mlflow.set_tracking_uri(DEFAULT_SERVER_URI)
        mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)

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
                    model_clone_cv = clone(optimized_model)
                    model_clone_cv.fit(
                        X_train, y_train, categorical_feature=cat_columns
                    )
                    y_pred = pd.Series(
                        model_clone_cv.predict(X_test), index=y_test.index
                    )
                    mae_cv = mean_absolute_error(y_test, y_pred)
                    mae_scores_this_split.append(mae_cv)

                avg_sales_this_series_cv = np.round(np.mean(data["item_sales"]), 2)
                mae_this_series_cv = np.round(
                    np.mean(np.array(mae_scores_this_split)), 2
                )
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
                model_clone_test = clone(optimized_model)

                model_clone_test.fit(X_train, y_train, categorical_feature=cat_columns)
                y_pred = pd.Series(model_clone_test.predict(X_test), index=y_test.index)

                mae_this_series_test = np.round(mean_absolute_error(y_test, y_pred), 2)
                avg_sales_this_series_test = np.round(np.mean(data["item_sales"]), 2)
                wmape_percentage_this_series_test = np.round(
                    100 * mae_this_series_test / (avg_sales_this_series_test + EPSILON),
                    2,
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

                metrics = {
                    "MAE CV": mae_this_series_cv,
                    "Mean sales CV": avg_sales_this_series_cv,
                    "WMAPE in percentage CV": wmape_percentage_this_series_cv,
                    "MAE Test": mae_this_series_test,
                    "Mean sales Test": avg_sales_this_series_test,
                    "WMAPE in percentage Test": wmape_percentage_this_series_test,
                }
                tags = {
                    "Model": "LGBMRegressor",
                    "Experiment": DEFAULT_EXPERIMENT_NAME,
                    "Store number": store_num,
                    "Item family": item_family,
                }
                with mlflow.start_run(run_name=f"Model for {(store_num, item_family)}"):
                    mlflow.lightgbm.log_model(
                        model_clone_test,
                        artifact_path=f"Models/{(store_num, item_family)}",
                    )
                    mlflow.log_params(best_params)
                    mlflow.log_metrics(metrics)
                    mlflow.log_text(
                        daily_metrics_test[(store_num, item_family)].to_json(
                            orient="split"
                        ),
                        f"Models/{(store_num, item_family)}/daily_metrics.json",
                    )
                    mlflow.set_tags(tags)

        save_metrics_and_models(
            metrics_cv_file,
            metrics_test_file,
            daily_metrics_file,
            models_file,
            mae_scores_cv,
            avg_sales_cv,
            wmape_percentage_scores_cv,
            mae_scores_test,
            avg_sales_test,
            wmape_percentage_scores_test,
            daily_metrics_test,
            models,
        )


if __name__ == "__main__":
    save_and_log_data()
