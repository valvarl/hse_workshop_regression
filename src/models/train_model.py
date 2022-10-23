# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.utils import save_as_pickle
import pandas as pd


from pipelines import train_model


@click.command()
@click.argument('input_train_data', type=click.Path(exists=True))
@click.argument('input_train_target', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path())
def main(input_train_data, input_train_target, output_model_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('run modeling pipeline')

    train_model.train(input_train_data, input_train_target, output_model_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
