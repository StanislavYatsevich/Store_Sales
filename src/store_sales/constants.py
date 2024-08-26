from pathlib import Path

NOT_HOLIDAY_DAY = "Not holiday"
NUMBER_OF_DAYS_TO_PREDICT = 15


DATA_FOR_STREAMLIT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "prepared_data" / "train_data.csv"

RAW_DATA_FOLDER_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw_data"
PREPARED_FOR_EDA_DATA_FOLDER_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "prepared_for_eda_data"
PREPARED_FINAL_DATA_FOLDER_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "prepared_final_data"

METRICS_FOLDER_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models_and_metrics"
MODELS_FOLDER_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models_and_metrics"

SERVER_URI = "http://127.0.0.1:8080"
DEFAULT_EXPERIMENT_NAME = "Cross validation models experiment"