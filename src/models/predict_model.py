import pandas as pd
import joblib
from src.config import *
from src.models.functions import metrics_print_for_catboost, metrics_print_for_lightgbm

X_test = pd.read_pickle(X_test_path)
Y_test = pd.read_pickle(Y_test_path)


# Predict for first model
model_catboost = joblib.load(model_catboost_path)

y_predict = model_catboost.predict(X_test)
y_predict_proba = model_catboost.predict_proba(X_test)

metrics_print_for_catboost(y_predict, y_predict_proba, Y_test, 'micro')
metrics_print_for_catboost(y_predict, y_predict_proba, Y_test, 'samples')

# Predict for second model
model_lgbm = joblib.load(model_lgbm_path)

y_predict = model_lgbm.predict(X_test)
y_predict_proba = model_lgbm.predict_proba(X_test)

metrics_print_for_lightgbm(y_predict, y_predict_proba, Y_test, 'micro')
metrics_print_for_lightgbm(y_predict, y_predict_proba, Y_test, 'samples')
