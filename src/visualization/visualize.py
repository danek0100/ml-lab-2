import os
import csv
import json


if os.path.exists('reports/exp_r2_catboost_results.json'):
    with open('reports/exp_r2_catboost_results.json', 'r') as f:
        data = json.load(f)

    with open('reports/exp_r2_dynamic_catboost.csv', mode='w', encoding='utf-8') as w_file:
        column_names = ["id", "result"]
        file_writer = csv.DictWriter(w_file, delimiter=",",
                                     lineterminator="\r", fieldnames=column_names)
        file_writer.writeheader()
        i = 0
        for value in data.values():
            file_writer.writerow(
                {"id": i, "result": value})
            i += 1
else:
    with open('reports/exp_r2_dynamic_catboost.csv', mode='w', encoding='utf-8') as w_file:
        print("No data")


if os.path.exists('reports/exp_r2_xgboost_results.json'):
    with open('reports/exp_r2_xgboost_results.json', 'r') as f:
        data = json.load(f)

    with open('reports/exp_r2_dynamic_xgboost.csv', mode='w', encoding='utf-8') as w_file:
        column_names = ["id", "result"]
        file_writer = csv.DictWriter(w_file, delimiter=",",
                                     lineterminator="\r", fieldnames=column_names)
        file_writer.writeheader()
        i = 0
        for value in data.values():
            file_writer.writerow(
                {"id": i, "result": value})
            i += 1
else:
    with open('reports/exp_r2_dynamic_xgboost.csv', mode='w', encoding='utf-8') as w_file:
        print("No data")
