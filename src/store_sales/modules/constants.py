import pandas as pd
from pathlib import Path


NOT_HOLIDAY_DAY = "Not holiday"
NUMBER_OF_DAYS_TO_PREDICT = 15

POPULAR_STORES = [3, 8, 11, 44, 45, 46, 47, 48, 49, 50, 51]
POPULAR_CLUSTERS = [5, 8, 11, 14, 17]
NON_POPULAR_CLUSTERS = [7]
SPECIAL_NON_WORKING_DAYS = ["Additional", "Bridge", "Transfer", "Event"]
POPULAR_HOLIDAYS = ["National"]
POPULAR_STATES = ["Pichincha"]
NON_POPULAR_STATES = ["Manabi", "Pastaza"]
POPULAR_CITIES = ["Quito", "Cayambe"]
NON_POPULAR_CITIES = ["Manta", "Puyo"]
POPULAR_STORE_TYPES = ["A"]
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
