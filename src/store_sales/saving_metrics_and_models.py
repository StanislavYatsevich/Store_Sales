import pickle
import json
from pathlib import Path
import xgboost as xgb
import pandas as pd
from store_sales import get_models_and_metrics_cross_validation, PREPARED_DATA_STAGE_2_FOLDER_PATH, METRICS_FOLDER_PATH, MODELS_FOLDER_PATH
import click

data = pd.read_csv(PREPARED_DATA_STAGE_2_FOLDER_PATH / "prepared_data.csv")
number_of_days_to_predict = 15
min_test_date = pd.to_datetime(data['date'].unique()[-number_of_days_to_predict])
train_data = data[pd.to_datetime(data['date']) < min_test_date]

params = {
    'max_depth': 2,
    'n_estimators': 259,
    'learning_rate': 0.016911596898275656,
    'subsample': 0.966503084641785,
    'colsample_bytree': 0.7652975950602404,
    'gamma': 0.3662258875815241,
    'reg_lambda': 2.254417098310434e-07,
    'alpha': 0.5897781687350846,
    'n_jobs': -1,
    'random_state': 42
}

xgboost = xgb.XGBRegressor(**params)
mae_scores, avg_sales, wmape_percentage_scores, models = get_models_and_metrics_cross_validation(train_data, xgboost)

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
    type = click.Path(),
    help="Path to the folder where models are saved to",
)

def save_data(metrics_folder_path, models_folder_path):
    metrics_file = Path(metrics_folder_path) / "metrics.json"
    models_file = Path(models_folder_path) / "models.pkl"

    metrics = {
        "MAE scores" : {str(k): v for k, v in mae_scores.items()},
        "Mean sales" : {str(k): v for k, v in avg_sales.items()},
        "WMAPE scores, %" : {str(k): v for k, v in wmape_percentage_scores.items()},
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics are successfully saved to {metrics_file}")

    with open(models_file, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models are successfully saved to {models_file}")


if __name__ == "__main__":
    save_data()
