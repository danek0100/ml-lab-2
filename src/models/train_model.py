import joblib
import category_encoders as ce
import lightgbm as ltb

from sklearn.preprocessing import *
from sklearn.compose import *
from sklearn.pipeline import *
from sklearn.multioutput import *
from catboost import CatBoostClassifier

from src.models.functions import *
from src.config import *
from src.utils import save_as_pickle

train = pd.read_pickle(processed_data_for_train_pkl)
target = pd.read_pickle(target_data_train_pkl)

X_train, Y_train, X_test, Y_test = split_data(train, target, test_size=test_size)

save_as_pickle(X_test, X_test_path)
save_as_pickle(Y_test, Y_test_path)

# Train first model
model = CatBoostClassifier(iterations=iterations, loss_function=loss_function,
                           eval_metric=eval_metric, learning_rate=learning_rate,
                           bootstrap_type=bootstrap_type, boost_from_average=boost_from_average,
                           ctr_leaf_count_limit=ctr_leaf_count_limit,
                           leaf_estimation_iterations=leaf_estimation_iterations,
                           leaf_estimation_method=leaf_estimation_method,
                           cat_features=CATEGORIES_COL_AFTER_PREP)

model_cast = MultiOutputClassifier(model)

pipeline_catboost = Pipeline([('model_cast', model_cast)])
pipeline_catboost.fit(X_train, Y_train)

joblib.dump(pipeline_catboost, model_catboost_path)


# Train second model
real_pipe = Pipeline([('scaler', StandardScaler())])

cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocess_pipe = ColumnTransformer(transformers=[
    ('real_cols', real_pipe, REAL_COLS),
    ('cat_cols', cat_pipe, CATEGORIES_COL_AFTER_PREP),
    ('cat_boost_cols', ce.CountEncoder(), CATEGORIES_COL_AFTER_PREP),  # for work with cat_features
]
)

model_ltb = ltb.LGBMClassifier(
    boosting_type='dart',
    num_leaves=20,
    learning_rate=0.01,
    n_estimators=500
)

model_pipe = Pipeline([('preprocess', preprocess_pipe), ('model', model_ltb)])

pipeline_ltb = MultiOutputClassifier(model_pipe, n_jobs=4)
pipeline_ltb.fit(X_train, Y_train)

joblib.dump(pipeline_ltb, model_lgbm_path)

