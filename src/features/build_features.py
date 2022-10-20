import pandas as pd
import logging
from src.features.functions import feature_generation
from src.config import *
from src.utils import save_as_pickle


def main():
    logger = logging.getLogger(__name__)
    logger.info('Feature Generation Target:')

    logger.log(logging.INFO, "Reading interim data")
    train = pd.read_pickle(interim_data_for_train_pkl)
    test = pd.read_pickle(interim_val_pkl)
    logger.log(logging.INFO, "Data was read")

    logger.log(logging.INFO, "Feature generation process started")
    train = feature_generation(train)
    test = feature_generation(test)
    logger.log(logging.INFO, "Feature generation process finished")

    logger.log(logging.INFO, "Results saving")
    save_as_pickle(test, processed_test_pkl)
    save_as_pickle(train, processed_data_for_train_pkl)
    logger.log(logging.INFO, "Results saved")
    logger.log(logging.INFO, "Feature generation target finished!")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
