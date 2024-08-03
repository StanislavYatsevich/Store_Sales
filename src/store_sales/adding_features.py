import pandas as pd
from store_sales import add_features, PREPARED_DATA_STAGE_1_FOLDER_PATH, PREPARED_DATA_STAGE_2_FOLDER_PATH
from pathlib import Path
import click


@click.command()
@click.option(
    "--input_data_folder_path",
    default=PREPARED_DATA_STAGE_1_FOLDER_PATH,
    type=click.Path(exists=True),
    help="Path to the folder with input data files",
)
@click.option(
    "--processed_data_folder_path",
    default=PREPARED_DATA_STAGE_2_FOLDER_PATH,
    type=click.Path(),
    help="Path to the folder where processed files are saved to",
)
def adding_features(input_data_folder_path, processed_data_folder_path):
    train_data = pd.read_csv(Path(input_data_folder_path) / "train_data.csv")
    valid_data = pd.read_csv(Path(input_data_folder_path) / "valid_data.csv")
    test_data = pd.read_csv(Path(input_data_folder_path) / "test_data.csv")
    data = pd.concat([train_data, valid_data, test_data], axis=0)
    data = add_features(data)
    data.to_csv(Path(processed_data_folder_path) / "prepared_data.csv", index=False)


if __name__ == "__main__":
    adding_features()
