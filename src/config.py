import yaml
import os

print(os.getcwd())

with open('params.yaml', encoding='utf-8') as conf_file:
    config = yaml.safe_load(conf_file)

# Data load
train_csv = config['data_load']['train_csv']
test_csv = config['data_load']['test_csv']

# Pre-process category
TARGET_COL = config['preprocess_category']['TARGET_COL']
ID_COL = config['preprocess_category']['ID_COL']
CAT_COLS = config['preprocess_category']['CAT_COLS']
REAL_COLS = config['preprocess_category']['REAL_COLS']
DATA_COLS = config['preprocess_category']['DATA_COLS']

BASEMENT_COLS = config['preprocess_category']['BASEMENT_COLS']
MASONRY_COLS = config['preprocess_category']['MASONRY_COLS']
GARAGE_COLS = config['preprocess_category']['GARAGE_COLS']
GARAGE_COL = config['preprocess_category']['GARAGE_COL']

MAX_NUMBER_NAN_VALUE_COLS = config['preprocess_category']['MAX_NUMBER_NAN_VALUE_COLS']
NUMBER_OF_TEST = config['preprocess_category']['NUMBER_OF_TEST']

# Data pre-process output
target_data_train_pkl = config['processed_paths']['target_data_train_pkl']
interim_data_for_train_pkl = config['interim_paths']['data_for_train_pkl']
interim_val_pkl = config['interim_paths']['val_pkl']

# Feature generation output
processed_test_pkl = config['processed_paths']['processed_test_pkl']
processed_data_for_train_pkl = config['processed_paths']['processed_data_for_train_pkl']

# Training
test_size = config['train']['test_size']
iterations = config['train']['iterations']
loss_function = config['train']['loss_function']
learning_rate = config['train']['learning_rate']
ctr_leaf_count_limit = config['train']['ctr_leaf_count_limit']
max_depth = config['train']['max_depth']

booster = config['train']['booster']
eta = config['train']['eta']
tree_method = config['train']['tree_method']

random_state = config['train']['random_state']

model_catboost_path = config['train']['model_catboost_path']
model_xgboost_path = config['train']['model_xgboost_path']
X_test_path = config['train']['X_test_path']
Y_test_path = config['train']['Y_test_path']

# Evaluate
score_metric = config['evaluate']['score_metric']
score_path_catboost = config['evaluate']['score_path_catboost']
score_path_xgboost = config['evaluate']['score_path_xgboost']
