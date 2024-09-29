import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from typing import Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from store_sales.modules import N_SPLITS


def optimize_lgb_params_with_optuna(
    train_data: pd.DataFrame, shops_number: int, n_trials: int
) -> Tuple[lgb.LGBMRegressor, dict]:
    """Optimizes hyperparameters of a given LGBMRegressor.

    Finds optimal hyperparameters for an LGBMRegressor within set boundaries
    for a given number of shops and trials. The optimization criteria is
    Mean Absolute Error (MAE) calculated using the cross-validation technique.

    Args:
        train_data: pd.DataFrame instance representing the train part of the data.
        shops_number: number of shops (<= 54 since there are exactly 54 shops in the dataset)
        to be randomly chosen for finding optimal hyperparameters.
        n_trials: number of trials for finding optimal hyperparameters.

    Returns:
        Tuple(best_model, best_params) where best_params is the dictionary
        representing the hyperparameters grid and best_model is the optimal
        model.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    random_stores = np.random.choice(
        train_data["store_number"].unique(), shops_number, replace=False
    )

    cat_columns = [
        col for col in train_data.columns if train_data[col].dtype == "object"
    ]
    for col in cat_columns:
        train_data[col] = train_data[col].astype("category")

    def objective(trial) -> float:
        parameters = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 0.5),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
        }
        model = lgb.LGBMRegressor(**parameters, random_state=42, n_jobs=-1, verbose=-1)
        mae_scores = []
        for store_num in random_stores:
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
                    model_clone = clone(model)
                    model_clone.fit(X_train, y_train, categorical_feature=cat_columns)
                    y_pred = pd.Series(model_clone.predict(X_test), index=y_test.index)
                    mae_cv = mean_absolute_error(y_test, y_pred)
                    mae_scores_this_split.append(mae_cv)

        return np.mean(np.array(mae_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = lgb.LGBMRegressor(
        **best_params, random_state=42, n_jobs=-1, verbose=-1
    )

    return best_model, best_params
