import numpy as np
import pandas as pd
import json
import skmultilearn
import csv

from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score
from src.config import *


def metrics_print_for_catboost(y_predict, y_predict_probability, y_true, average):
    metric_result = {'average': average,
                     'precision': precision_score(y_true, y_predict, average=average),
                     'recall': recall_score(y_true, y_predict, average=average),
                     'f1': f1_score(y_true, y_predict, average=average),
                     }
    print(average)
    print('precision ', metric_result['precision'])
    print('recall ', metric_result['recall'])
    print('f1 ', metric_result['f1'])

    if average == 'micro':
        y_predict_probability = np.transpose([predict[:, 1] for predict in y_predict_probability])
        metric_result['ROCAUC'] = roc_auc_score(y_true, y_predict_probability, average=average)
        print('ROCAUC', metric_result['ROCAUC'])
    else:
        y_predict_probability = np.transpose([predict[:, 1] for predict in y_predict_probability])
        ROCAUC_score = roc_auc_score(y_true, y_predict_probability, average=None)
        print('ROCAUC', ROCAUC_score)
        csv_write(score_path_rocauc_samples_catboost, ROCAUC_score, TARGET_COLS)

    print('\n')

    with open(score_path_catboost, 'w') as f:
        json.dump(metric_result, f)


def metrics_print_for_lightgbm(y_predict, y_predict_probability, y_true, average):
    metric_result = {'average': average,
                     'precision': precision_score(y_true, y_predict, average=average),
                     'recall': recall_score(y_true, y_predict, average=average),
                     'f1': f1_score(y_true, y_predict, average=average),
                     }
    print(average)
    print('precision ', metric_result['precision'])
    print('recall ', metric_result['recall'])
    print('f1 ', metric_result['f1'])

    if average == 'micro':
        y_predict_probability = np.transpose([predict[:, 1] for predict in y_predict_probability])
        metric_result['ROCAUC'] = roc_auc_score(y_true, y_predict_probability, average=average)
        print('ROCAUC', metric_result['ROCAUC'])
        with open(score_path_rocauc_samples_lgbm, 'w') as f:
            json.dump(metric_result, f)

    else:
        y_predict_probability = np.transpose([predict[:, 1] for predict in y_predict_probability])
        ROCAUC_score = roc_auc_score(y_true, y_predict_probability, average=None)
        print('ROCAUC', ROCAUC_score)
        csv_write(score_path_rocauc_samples_lgbm, ROCAUC_score, TARGET_COLS)
    print('\n')

    with open(score_path_lightgbm, 'w') as f:
        json.dump(metric_result, f)


def turning_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in CATEGORIES_COL_AFTER_PREP:
            df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype(np.int8)
    return df


# Метрики несбалансированные, при рабиение на train\test мы должны учитывать соотношение 0 и 1.
# Данная функция позволяет работать со случаем multi-labels.
def split_data(data, target, test_size):
    # Было б все легко, если б не так все печально хаха!!!!!!!!
    # Данный встроенный метод работает только со списками, поэтому пришлось выкручиваться :)
    X_train, Y_train, X_test, Y_test = skmultilearn.model_selection.iterative_stratification.iterative_train_test_split(
        np.array(data), np.array(target), test_size=test_size)

    X_train = pd.DataFrame(X_train, columns=list(COL))
    X_test = pd.DataFrame(X_test, columns=list(COL))
    Y_train = pd.DataFrame(Y_train, columns=list(TARGET_COLS))
    Y_test = pd.DataFrame(Y_test, columns=list(TARGET_COLS))

    X_train = turning_types(X_train)
    X_test = turning_types(X_test)

    return X_train, Y_train, X_test, Y_test


def csv_write(path, data, columns):
    with open(path, mode='w', encoding='utf-8') as w_file:
        column_names = ["column", "result"]
        file_writer = csv.DictWriter(w_file, delimiter=",",
                                     lineterminator="\r", fieldnames=column_names)
        file_writer.writeheader()
        for i in range(len(data)):
            file_writer.writerow(
#                {"column": columns[i], "result": str(data[i])})
                {"column": i, "result": str(data[i])})
