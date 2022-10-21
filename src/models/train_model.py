import joblib
import logging
import pandas as pd

from sklearn.preprocessing import *
from sklearn.compose import *
from sklearn.pipeline import *
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor
import category_encoders as ce
import xgboost

from src.config import *
from src.utils import save_as_pickle


def main():
    logger = logging.getLogger(__name__)
    logger.info('Train Target:')

    logger.log(logging.INFO, "Starting reading data from pkl...")
    train = pd.read_pickle(processed_data_for_train_pkl)
    target = pd.read_pickle(target_data_train_pkl)
    logger.log(logging.INFO, "Data successfully read")

    logger.log(logging.INFO, "Start data split process")
    X_train, X_test, Y_train, Y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    logger.log(logging.INFO, "Data splited")

    logger.log(logging.INFO, "Selections saving")
    save_as_pickle(X_test, X_test_path)
    save_as_pickle(Y_test, Y_test_path)
    logger.log(logging.INFO, "Selections saved")

    # Train CatBoost model
    logger.log(logging.INFO, "Starting CatBoost train")
    model = CatBoostRegressor(
        iterations=iterations,
        max_depth=max_depth,
        ctr_leaf_count_limit=ctr_leaf_count_limit,
        loss_function=loss_function,
        learning_rate=learning_rate,
        cat_features=CAT_COLS)

    pipeline_catboost = Pipeline([('model_cast', model)])
    pipeline_catboost.fit(X_train, Y_train)
    joblib.dump(pipeline_catboost, model_catboost_path)
    logger.log(logging.INFO, "CatBoost was trained and model was saved")

    # Train xgboost model
    logger.log(logging.INFO, "Starting XGBoost train")
    real_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    pre_process_pipe = ColumnTransformer(transformers=[
        ('real_cols', real_pipe, REAL_COLS),
        ('cat_cols', cat_pipe, CAT_COLS),
        ('cat_bost_cols', ce.CountEncoder(), CAT_COLS),
    ]
    )

    model = xgboost.XGBRegressor(
        booster=booster,
        eta=eta,
        tree_method=tree_method
    )
    model_pipe = Pipeline([('preprocess', pre_process_pipe), ('model', model)])
    pipeline_xg = model_pipe
    pipeline_xg.fit(X_train, Y_train)
    joblib.dump(pipeline_xg, model_xgboost_path)
    logger.log(logging.INFO, "XGBoost was trained and model was saved")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
