import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import click
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from pathlib import Path
from store_sales import (
    init_mlflow_experiment,
    encode_features,
    get_models_and_metrics_cross_validation,
    save_metrics_and_models,
    optimize_xgboost_params_with_optuna,
    PREPARED_DATA_STAGE_2_FOLDER_PATH,
    METRICS_FOLDER_PATH,
    MODELS_FOLDER_PATH,
    SERVER_URI,
    DEFAULT_EXPERIMENT_NAME,
)


data = pd.read_csv(PREPARED_DATA_STAGE_2_FOLDER_PATH / "prepared_data.csv")
number_of_days_to_predict = 15
min_test_date = pd.to_datetime(data["date"].unique()[-number_of_days_to_predict])
train_data = data[pd.to_datetime(data["date"]) < min_test_date]

optimized_model, best_params = optimize_xgboost_params_with_optuna(train_data, 5, 100)

tscv = TimeSeriesSplit(n_splits=5)

server_uri = SERVER_URI
experiment_name = DEFAULT_EXPERIMENT_NAME


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
    help="To send metrics and models to MLFlow server or not",
)
def save_and_log_data(metrics_folder_path, models_folder_path, send_to_server):
    metrics_folder_path.mkdir(parents=True, exist_ok=True)
    models_folder_path.mkdir(parents=True, exist_ok=True)

    metrics_file = Path(metrics_folder_path) / "metrics.json"
    models_file = Path(models_folder_path) / "models.pkl"

    if not send_to_server:
        mae_scores, avg_sales, wmape_percentage_scores, models = (
            get_models_and_metrics_cross_validation(train_data, optimized_model)
        )
        save_metrics_and_models(
            metrics_file,
            models_file,
            mae_scores,
            avg_sales,
            wmape_percentage_scores,
            models,
        )
    else:
        mae_scores = dict()
        avg_sales = dict()
        wmape_percentage_scores = dict()
        models = dict()

        init_mlflow_experiment(server_uri, experiment_name)

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
                    model_clone = clone(optimized_model)
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

                metrics = {
                    "MAE": mae_this_series,
                    "Mean sales": avg_sales_this_series,
                    "WMAPE_in_percentage": wmape_percentage_this_series,
                }
                tags = {
                    "Model": "XGBRegressor",
                    "Experiment": experiment_name,
                    "Store number": store_num,
                    "Item family": item_family,
                }
                with mlflow.start_run(run_name=f"Model for {(store_num, item_family)}"):
                    mlflow.xgboost.log_model(
                        model_clone, artifact_path=f"Models/{(store_num, item_family)}"
                    )
                    mlflow.log_params(best_params)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tags(tags)

        save_metrics_and_models(
            metrics_file,
            models_file,
            mae_scores,
            avg_sales,
            wmape_percentage_scores,
            models,
        )


if __name__ == "__main__":
    save_and_log_data()
