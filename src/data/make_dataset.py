# -*- coding: utf-8 -*-
import click
import logging
import os
import pandas as pd

from pathlib import Path
from src.data.preprocess import extract_target, pre_process_target, pre_process_df, pre_process_val
from src.config import *
from src.utils import save_as_pickle


@click.command()
@click.argument('output_target_filepath', type=click.Path())
def main(output_target_filepath=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Make DataSet Target:')

    logger.log(logging.INFO, "Starting reading data from csv...")
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    logger.log(logging.INFO, "Data successfully read")

    logger.log(logging.INFO, "Starting pre-processing for train/val data")
    train = pre_process_df(train)
    test = pre_process_val(test)
    logger.log(logging.INFO, "Pre-processing finished")

    # train
    logger.log(logging.INFO, "Starting pre-processing for target")
    train, target = extract_target(train)
    target = pre_process_target(target)
    logger.log(logging.INFO, "Pre-processing for target finished")
    logger.log(logging.INFO, "Results saving")
    save_as_pickle(target, target_data_train_pkl)
    save_as_pickle(train, interim_data_for_train_pkl)

    # val
    save_as_pickle(test, interim_val_pkl)
    logger.log(logging.INFO, "Results saved")
    logger.log(logging.INFO, "Pre-processing target finished!")


    # if output_target_filepath:
    #     train, target = extract_target(train)
    #     target = preprocess_target(target)
    #     save_as_pickle(target, target_data_train_pkl)
    #
    # save_as_pickle(train, interim_data_for_train_pkl)
    # save_as_pickle(test, interim_test_pkl)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
