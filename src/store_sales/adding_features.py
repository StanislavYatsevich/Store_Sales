import pandas as pd
from store_sales import add_features
import click

@click.command()
@click.option('--input_data_folder', required=True, type=click.Path(exists=True), help='Path to the folder with input data files')
@click.option('--output_data_folder', required=True, type=click.Path(), help='Path to the folder where processed files are saved to')

def adding_features(input_data_folder, output_data_folder):
    train_data = pd.read_csv(f'{input_data_folder}/train_data.csv')
    valid_data = pd.read_csv(f'{input_data_folder}/valid_data.csv')
    test_data = pd.read_csv(f'{input_data_folder}/test_data.csv')

    data = pd.concat([train_data, valid_data, test_data], axis=0)
    data = add_features(data)
    data.to_csv(f'{output_data_folder}/prepared_data.csv', index=False)

if __name__ == '__main__':
    adding_features()