import pandas as pd
import click
from pathlib import Path
from store_sales.modules import (
    add_features,
    PREPARED_FOR_EDA_DATA_FOLDER_PATH,
    PREPARED_FINAL_DATA_FOLDER_PATH,
)


@click.command()
@click.option(
    "--input_data_folder_path",
    default=PREPARED_FOR_EDA_DATA_FOLDER_PATH,
    type=click.Path(exists=True),
    help="Path to the folder with input data files",
)
@click.option(
    "--prepared_final_data_folder_path",
    default=PREPARED_FINAL_DATA_FOLDER_PATH,
    type=click.Path(),
    help="Path to the folder where finally prepared data files are saved to",
)
def add_new_features(input_data_folder_path, prepared_final_data_folder_path):
    prepared_final_data_folder_path = Path(prepared_final_data_folder_path)
    prepared_final_data_folder_path.mkdir(parents=True, exist_ok=True)
    train_data = pd.read_csv(Path(input_data_folder_path) / "train_data.csv")
    test_data = pd.read_csv(Path(input_data_folder_path) / "test_data.csv")

    train_data, test_data = add_features(train_data, test_data)

    train_data.to_csv(
        Path(prepared_final_data_folder_path) / "train_data.csv", index=False
    )
    test_data.to_csv(
        Path(prepared_final_data_folder_path) / "test_data.csv", index=False
    )


if __name__ == "__main__":
    add_new_features()
