from pathlib import Path

NOT_HOLIDAY_DAY = "Not holiday"
NUMBER_OF_DAYS_TO_PREDICT = 15


DATA_FOR_STREAMLIT_PATH = Path("../../data/prepared_data/train_data.csv")

RAW_DATA_FOLDER_PATH = Path("../../data/raw_data")
PREPARED_DATA_STAGE_1_FOLDER_PATH = Path("../../data/prepared_data")
PREPARED_DATA_STAGE_2_FOLDER_PATH = Path("../../data/prepared_data")

METRICS_FOLDER_PATH = Path("../../data/models_and_metrics")
MODELS_FOLDER_PATH = Path("../../data/models_and_metrics")

SERVER_URI = "http://127.0.0.1:8080"
DEFAULT_EXPERIMENT_NAME = "Cross validation models experiment"