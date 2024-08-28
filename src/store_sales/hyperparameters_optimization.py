import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from typing import Tuple
from sklearn.model_selection import TimeSeriesSplit
from store_sales import encode_features, get_mae, N_SPLITS


def optimize_xgboost_params_with_optuna(
    train_data: pd.DataFrame, shops_number: int, n_trials: int
) -> Tuple[xgb.XGBRegressor, dict]:
    """Optimizes hyperparameters of a given XGBRegressor.

    Finds optimal hyperparameters for an XGBRegressor within set boundaries
    for a given number of shops and trials. The optimization criteria is
    Mean Absolute Error (MAE) calculated using the cross-validation technique.

    Args:
        train_data: pd.DataFrame instance representing the train part of the data.
        shops_number: number of shops (<= 54) to be randomly chosen for finding
            optimal hyperparameters.
        n_trials: number of trials for finding optimal hyperparameters.

    Returns:
        A Tuple(best_model, best_params) where best_params is the dictionary
        representing the hyperparameters grid and best_model is the optimal
        model.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
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

        return np.mean(np.array(mae_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)

    return best_model, best_params