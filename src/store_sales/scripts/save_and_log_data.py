import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import click
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from pathlib import Path
from store_sales.modules import (
    encode_features,
    get_mae,
    get_models_and_metrics_cv_and_testing,
    save_metrics_and_models,
    optimize_xgboost_params_with_optuna,
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
    models_file = Path(models_folder_path) / "models.pkl"

    train_data = pd.read_csv(PREPARED_FINAL_DATA_FOLDER_PATH / "train_data.csv")
    test_data = pd.read_csv(PREPARED_FINAL_DATA_FOLDER_PATH / "test_data.csv")

    optimized_model, best_params = optimize_xgboost_params_with_optuna(
        train_data, N_SHOPS_OPTUNA, N_TRIALS_OPTUNA
    )
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    if not send_to_server:
        (
            mae_scores_cv,
            avg_sales_cv,
            wmape_percentage_scores_cv,
            mae_scores_test,
            avg_sales_test,
            wmape_percentage_scores_test,
            models,
        ) = get_models_and_metrics_cv_and_testing(
            train_data, test_data, optimized_model
        )
        save_metrics_and_models(
            metrics_cv_file,
            metrics_test_file,
            models_file,
            mae_scores_cv,
            avg_sales_cv,
            wmape_percentage_scores_cv,
            mae_scores_test,
            avg_sales_test,
            wmape_percentage_scores_test,
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
                    model_clone_cv = clone(optimized_model)
                    mae_cv = get_mae(
                        X_train_encoded, X_test_encoded, y_train, y_test, model_clone_cv
                    )
                    mae_scores_this_split.append(mae_cv)

                avg_sales_this_series_cv = np.round(
                    np.mean(train_data["item_sales"]), 2
                )
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

                test_data = test_data[
                    (test_data["store_number"] == store_num)
                    & (test_data["item_family"] == item_family)
                ]
                X_train = train_data.drop(["item_sales"], axis=1)
                y_train = train_data["item_sales"]
                X_test = test_data.drop(["item_sales"], axis=1)
                y_test = test_data["item_sales"]
                X_train_encoded, X_test_encoded = encode_features(X_train, X_test)
                model_clone_test = clone(optimized_model)

                model_clone_test.fit(X_train_encoded, y_train)
                y_pred = pd.Series(
                    model_clone_test.predict(X_test_encoded), index=y_test.index
                )

                mae_this_series_test = np.round(mean_absolute_error(y_test, y_pred), 2)
                avg_sales_this_series_test = np.round(
                    np.mean(test_data["item_sales"]), 2
                )
                wmape_percentage_this_series_test = np.round(
                    100 * mae_this_series_test / (avg_sales_this_series_test + EPSILON),
                    2,
                )

                mae_scores_test[(store_num, item_family)] = mae_this_series_test
                avg_sales_test[(store_num, item_family)] = avg_sales_this_series_test
                wmape_percentage_scores_test[(store_num, item_family)] = (
                    wmape_percentage_this_series_test
                )

                models_fitted[(store_num, item_family)] = model_clone_test

                metrics = {
                    "MAE CV": mae_this_series_cv,
                    "Mean sales CV": avg_sales_this_series_cv,
                    "WMAPE in percentage CV": wmape_percentage_this_series_cv,
                    "MAE Test": mae_this_series_test,
                    "Mean sales Test": avg_sales_this_series_test,
                    "WMAPE in percentage Test": wmape_percentage_scores_test,
                }
                tags = {
                    "Model": "XGBRegressor",
                    "Experiment": DEFAULT_EXPERIMENT_NAME,
                    "Store number": store_num,
                    "Item family": item_family,
                }
                with mlflow.start_run(run_name=f"Model for {(store_num, item_family)}"):
                    mlflow.xgboost.log_model(
                        model_clone_test,
                        artifact_path=f"Models/{(store_num, item_family)}",
                    )
                    mlflow.log_params(best_params)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tags(tags)

        save_metrics_and_models(
            metrics_cv_file,
            metrics_test_file,
            models_file,
            mae_scores_cv,
            avg_sales_cv,
            wmape_percentage_scores_cv,
            mae_scores_test,
            avg_sales_test,
            wmape_percentage_scores_test,
            models,
        )


if __name__ == "__main__":
    save_and_log_data()
