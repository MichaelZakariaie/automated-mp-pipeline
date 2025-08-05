import insight_config
import lightgbm as lgb
from catboost import CatBoostClassifier
from flaml import AutoML
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

BASIC_MODELS = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "rf": RandomForestClassifier(),
    "ridge": RidgeClassifier(),
    "logreg": LogisticRegression(random_state=insight_config.RANDOM_SEED, C=1.0),
    "svc": SVC(gamma="auto"),
    "lgbm": None,  # uses lgb.train() instead
    "catboost": None,
    # NOTE: when adding models, ensure warm_start=False
    # TODO:
    # add 1NN-DTW, LGBM, XGBoost, CatBoost
    # add tsai methods...
    # catch22 and other feature methods; multirocket, tsfresh...
}


def init_lgb_params():
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",  # 'dart', 'goss'
        "metric": "auc",  # ["auc", "binary_logloss", "f1", "accuracy", "mcc"]
        # "categorical_feature": "name:task_id_renamed",  # TODO: add
        "n_estimators": 1000,  # 100-1500
        "num_leaves": 10,  # 4, 8, 16, 32, 64
        "min_child_samples": 25,  # 20-100
        "learning_rate": 0.05,  # .01, .05, 0.1
        "max_bin": 100,  # 64-256
        "reg_alpha": 0.0,  # 0.001-1.0 for this dataset # L1
        "reg_lambda": 0.0,  # 0.1-3.0 for this dataset # L2
        "verbosity": -1,
        "early_stopping_round": 30,  # 10-100
        # "feature_fraction": 0.8,  # 0.5-0.8 # decrease to reduce train time
        # "bagging_fraction": 0.8,  # 0.6-0.8 # decrease to reduce train time
        # "max_depth": 6,  # 3-6
        "n_jobs": -1,  # use all cores
    }
    lgb_clf = lgb.LGBMClassifier(**params)  # TODO: legacy, remove placeholder
    return lgb_clf, params


def init_catboost_params():
    params = {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": insight_config.RANDOM_SEED,
        "eval_metric": "AUC",
        "early_stopping_rounds": 30,
        "verbose": False,
        "thread_count": -1,
    }
    catboost_clf = CatBoostClassifier(**params)
    return catboost_clf, params


def run_flaml(X_train, y_train):
    automl = AutoML()
    # automl.fit(X_train, y_train, task="classification")
    # automl.fit(X_train, y_train, task="classification", early_stop=True)
    # automl.fit(X_train, y_train, task="seq-classification")  # seq-?
    # automl.fit(X_train, y_train, task="classification", time_budget=-1)
    # automl.fit(X_train, y_train, task="classification", time_budget=-300, n_split=5)
    automl.fit(
        X_train,
        y_train,
        task="classification",
        time_budget=1200,
        metric="roc_auc",  # "f1",
        early_stop=True,
        log_file_name=".flaml-logs/flaml_experiment.log",
        # verbose=3 # default level, INFO
        # verbose=2,  # WARNING level
    )
    # automl.fit(X_train, y_train, task="classification", time_budget=300, metric="f1")
    return automl
