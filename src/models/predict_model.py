import pandas as pd
import joblib
import logging
from src.config import *
from src.models.functions import print_metric


def main():
    X_test = pd.read_pickle(X_test_path)
    Y_test = pd.read_pickle(Y_test_path)

    # Predict for CatBoost model
    catboost_model = joblib.load(model_catboost_path)

    y_predict = catboost_model.predict(X_test)
    print_metric(Y_test, y_predict, score_path_catboost)

    # Predict for XGBoost model
    xgboost_model = joblib.load(model_xgboost_path)

    y_predict = xgboost_model.predict(X_test)
    print_metric(Y_test, y_predict, score_path_xgboost)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

