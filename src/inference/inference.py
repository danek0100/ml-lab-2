# -*- coding: utf-8 -*-
import click
import logging
import joblib
import pandas as pd
import os
import csv

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.config import model_catboost_path, model_lgbm_path, TARGET_COLS
from src.data.preprocess import preprocess
from src.features.build_features import feature_generation


def csv_write(path, data):
    with open(path, mode='w', encoding='utf-8') as w_file:
        column_names = ['Index'] + TARGET_COLS
        file_writer = csv.DictWriter(w_file, delimiter=",",
                                     lineterminator="\r", fieldnames=column_names)
        file_writer.writeheader()
        for i in range(len(data)):
            struct = {"Index": i}
            for j in range(len(data[i])):
                struct[column_names[j+1]] = data[i][j]
            file_writer.writerow(struct)


@click.command()
@click.argument('data_filepath', type=click.Path())
def main(data_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Model inference for ' + data_filepath + '')

    data_filepath = data_filepath[14:]
    test = pd.read_csv(data_filepath)
    test = preprocess(test)
    test = feature_generation(test)

    model_catboost = joblib.load(model_catboost_path)
    model_lgbm = joblib.load(model_lgbm_path)

    y_predicted_castboost = model_catboost.predict(test)
    logger.log(logging.INFO, '\n' + str(y_predicted_castboost))
    csv_write('reports/' + data_filepath[:-4].replace('/', '_') + '_result_catboost.csv', y_predicted_castboost)

    y_predicted_lgbm = model_lgbm.predict(test)
    logger.log(logging.INFO, '\n' + str(y_predicted_lgbm))
    csv_write('reports/' + data_filepath[:-4].replace('/', '_') + '_result_lgbm.csv', y_predicted_lgbm)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

