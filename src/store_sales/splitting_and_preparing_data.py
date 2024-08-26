import pandas as pd
import click
from pathlib import Path
from store_sales import (
    prepare_data,
    RAW_DATA_FOLDER_PATH,
    PREPARED_FOR_EDA_DATA_FOLDER_PATH,
    NUMBER_OF_DAYS_TO_PREDICT
)


@click.command()
@click.option(
    "--raw_data_folder_path",
    default=RAW_DATA_FOLDER_PATH,
    type=click.Path(exists=True),
    help="Path to the folder with raw data files",
)
@click.option(
    "--prepared_for_eda_data_folder_path",
    default=PREPARED_FOR_EDA_DATA_FOLDER_PATH,
    type=click.Path(),
    help="Path to the folder where prepared for EDA data files are saved to",
)
def split_and_prepare_data(raw_data_folder_path, prepared_for_eda_data_folder_path):
    prepared_for_eda_data_folder_path.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(Path(raw_data_folder_path) / "train.csv")
    holidays_events_data = pd.read_csv(
        Path(raw_data_folder_path) / "holidays_events.csv"
    )
    oil_data = pd.read_csv(Path(raw_data_folder_path) / "oil.csv")
    stores_data = pd.read_csv(Path(raw_data_folder_path) / "stores.csv")

    data = prepare_data(data, holidays_events_data, oil_data, stores_data)

    min_test_date = pd.to_datetime(data["date"].unique()[-NUMBER_OF_DAYS_TO_PREDICT])
    train_data = data[pd.to_datetime(data["date"]) < min_test_date]
    test_data = data[pd.to_datetime(data["date"]) >= min_test_date]

    train_data.to_csv(Path(prepared_for_eda_data_folder_path) / "train_data.csv", index=False)
    test_data.to_csv(Path(prepared_for_eda_data_folder_path) / "test_data.csv", index=False)


if __name__ == "__main__":
    split_and_prepare_data()
