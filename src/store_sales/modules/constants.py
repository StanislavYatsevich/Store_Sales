import pandas as pd
from pathlib import Path


NOT_HOLIDAY_DAY = "Not holiday"
NUMBER_OF_DAYS_TO_PREDICT = 15
MIN_TRAIN_DATE = pd.to_datetime("2013-01-01")

SPECIAL_NON_WORKING_DAYS = ["Additional", "Bridge", "Transfer", "Event"]
POPULAR_HOLIDAYS = ["National"]
DATE_OF_EARTHQUAKE = pd.to_datetime("2016-04-16")

OIL_PRICE_FALLING_START_1 = pd.to_datetime("2014-07-01")
OIL_PRICE_FALLING_FINISH_1 = pd.to_datetime("2015-01-31")
OIL_PRICE_FALLING_START_2 = pd.to_datetime("2015-06-01")
OIL_PRICE_FALLING_FINISH_2 = pd.to_datetime("2016-02-29")

EPSILON = 10**-5
N_SPLITS = 5
N_SHOPS_OPTUNA = 5
N_TRIALS_OPTUNA = 100
LAGGED_FEATRUES_WINDOW_SIZE = 30

ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = ROOT_PATH / "data"

DATA_FOR_STREAMLIT_PATH = DATA_PATH / "prepared_for_eda_data" / "train_data.csv"
RAW_DATA_FOLDER_PATH = DATA_PATH / "raw_data"
PREPARED_FOR_EDA_DATA_FOLDER_PATH = DATA_PATH / "prepared_for_eda_data"
PREPARED_FINAL_DATA_FOLDER_PATH = DATA_PATH / "prepared_final_data"
METRICS_FOLDER_PATH = DATA_PATH / "models_and_metrics"
MODELS_FOLDER_PATH = DATA_PATH / "models_and_metrics"

DEFAULT_SERVER_URI = "http://127.0.0.1:8080"
DEFAULT_EXPERIMENT_NAME = "Cross validation models experiment"
