import json

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score


def print_metric(y_true, y_predict, file_path):
    metric_results = {'R2': r2_score(y_true, y_predict),
                      'MSE': mean_squared_error(y_true, y_predict),
                      'MAPE': mean_absolute_percentage_error(y_true, y_predict),
                      'EV': explained_variance_score(y_true, y_predict)}

    print()
    print('R2: ', metric_results['R2'])
    print('MSE: ', metric_results['MSE'])
    print('MAPE: ', metric_results['MAPE'])
    print('Explained variance', metric_results['EV'])
    print()

    with open(file_path, 'w') as f:
        json.dump(metric_results, f)
