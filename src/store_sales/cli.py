import click

@click.command()
@click.option('--input_path', required=True, type=click.Path(exists=True), help='Path to the input file')
@click.option('--output_path', required=True, type=click.Path(), help='Path to save the processed file')

"""def preprocess(input_path, output_path):
    preprocess_data(input_path, output_path)

    if __name__ == '__main__':
        preprocess()"""