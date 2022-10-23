# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from pipelines import preprocessing

processed_data_path = 'data/processed/'


@click.command()
@click.argument('input_train_path', type=click.Path(exists=True))
@click.argument('input_test_path', type=click.Path(exists=True))
@click.argument('output_dir', default=processed_data_path, type=click.Path())
def main(input_train_path, input_test_path, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    preprocessing.preprocess(input_train_path, input_test_path, output_dir)

    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
