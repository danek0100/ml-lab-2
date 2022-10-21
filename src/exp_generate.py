import os
import json
import csv


def csv_write(path, data):
    with open(path, mode='w', encoding='utf-8') as w_file:
        column_names = ["column", "result"]
        file_writer = csv.DictWriter(w_file, delimiter=",",
                                     lineterminator="\r", fieldnames=column_names)
        file_writer.writeheader()
        for i in range(len(data)):
            file_writer.writerow(
                #                {"column": columns[i], "result": str(data[i])})
                {"column": i, "result": str(data[i])})


# Many param was deleted after 128 test for generate plots :-)
# catboost params
test_sizes = [0.2]
iterations = [100]
loss_functions = ['RMSE']
learning_rates = [0.2]
ctr_leaf_count_limits = [5]
max_depths = [4]

# xgboost
boosters = ['dart']
etas = [0.1]
tree_methods = ['exact']

random_states = [2, 5, 99, 1312]

command_base = 'dvc exp run -f '
set_param = '--set-param '
new_command_test_size = ''
new_command_iteration = ''
new_command_loss_function = ''
new_command_learning_rate = ''
new_command_ctr_leaf_count_limit = ''
new_command_max_depth = ''
new_command_booster = ''
new_command_eta = ''
new_command_tree_method = ''
new_command_random_state = ''

r2_catboost = {}
r2_xgboost = {}
catboost = ''
xgboost = ''

experiments = []

with open('../dvc_output.txt', 'w') as f:
    f.write('')

os.chdir('..')

for test_size in test_sizes:
    new_command_test_size = '' + command_base + set_param + 'train.test_size=' + str(test_size) + ' '
    for iteration in iterations:
        new_command_iteration = new_command_test_size + set_param + 'train.iterations=' + str(iteration) + ' '
        for loss_function in loss_functions:
            new_command_loss_function = new_command_iteration + set_param + 'train.loss_function=' + str(
                loss_function) + ' '
            for learning_rate in learning_rates:
                new_command_learning_rate = new_command_loss_function + set_param + 'train.learning_rate=' + str(
                    learning_rate) + ' '
                for ctr_leaf_count_limit in ctr_leaf_count_limits:
                    new_command_ctr_leaf_count_limit = new_command_learning_rate + set_param + 'train.ctr_leaf_count_limit=' + str(
                        ctr_leaf_count_limit) + ' '
                    for max_depth in max_depths:
                        new_command_max_depth = new_command_ctr_leaf_count_limit + set_param + 'train.max_depth=' + str(
                            max_depth) + ' '
                        for booster in boosters:
                            new_command_booster = new_command_max_depth + set_param + 'train.booster=' + str(
                                booster) + ' '
                            for eta in etas:
                                new_command_eta = new_command_booster + set_param + 'train.eta=' + str(
                                    eta) + ' '
                                for tree_method in tree_methods:
                                    new_command_tree_method = new_command_eta + set_param + 'train.tree_method=' + str(
                                        tree_method) + ' '
                                    for random_state in random_states:
                                        with open('dvc_output.txt', 'w') as f:
                                            f.write('')
                                        new_command_random_state = new_command_tree_method + set_param + 'train' \
                                                                                                         '.random_state=' + str(
                                            random_state) + ' '
                                        new_command_random_state += '>> dvc_output.txt\n'

                                        os.system(new_command_random_state)

                                        with open('dvc_output.txt', 'r') as f:
                                            text = f.readlines()

                                        find = False
                                        for line in text:
                                            index = line.find('exp-')
                                            if index != -1:
                                                experiments.append(line[index:-1])
                                                r2_catboost[experiments[-1]] = catboost
                                                r2_xgboost[experiments[-1]] = xgboost
                                                os.system('dvc exp branch ' + experiments[-1] + ' ' + experiments[-1])
                                                break

                                            if not find:
                                                catboost_index = line.find('R2:  ')
                                                if catboost_index != -1:
                                                    catboost = line[4:-1]
                                                    find = True
                                            else:
                                                xgboost_index = line.find('R2:  ')
                                                if xgboost_index != -1:
                                                    xgboost = line[5:-1]
                                                    find = False

with open('reports/exp_r2_catboost_results.json', 'w') as f:
    json.dump(r2_catboost, f)

with open('reports/exp_r2_xgboost_results.json', 'w') as f:
    json.dump(r2_catboost, f)

catboost_values = []
xgboost_values = []
for value in r2_catboost.values():
    catboost_values.append(value)
for value in r2_xgboost.values():
    xgboost_values.append(value)

csv_write('reports/exp_r2_dynamic_catboost.csv', catboost_values)
csv_write('reports/exp_r2_dynamic_xgboost.csv', xgboost_values)
