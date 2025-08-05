# rocket + ridge, lgbm
import argparse
import json
import os
import time
from collections.abc import Iterable
from typing import Tuple

import awswrangler as wr
import insight_config
import insight_scorer
import lightgbm as lgb
import model_zoo
import numpy as np
import pandas as pd
import rockets
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier
from flaml.automl.data import get_output_from_log
from matplotlib import pyplot as plt
from scipy.signal import resample
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from wrangle_ts import dataset_stats, label_dist, print_dataset_stats

#######################################
#### Lower level


def get_fps_from_filename(fname):
    parts = fname.split("_")
    for part in parts:
        if part.endswith("fps"):
            fps_str = part.replace("fps", "")
            return int(fps_str)
    return None


def remove_intermediate_pcl(  # TODO extend/generalize
    df: pd.DataFrame, ptsd_threshold=33, buffer=8
) -> pd.DataFrame:
    """Remove people who scored near the PTSD threshold on PCL (Dr. Rothbaum).
    Might improve models.

    dx : pcl dataframe from a query
    """
    print("\nRemoving intermediate PCL scores...")
    # e.g. ptsd_threshold = 33, buffer = 8 => PCL (25, 41)
    dx = df.copy()
    dx = dx[
        ~dx["pcl_score"].between(
            ptsd_threshold - buffer, ptsd_threshold + buffer, inclusive="both"
        )
    ]
    return dx


def get_cv_splits(cv_type=insight_config.CV_TYPE, num_folds=insight_config.NUM_FOLDS):
    """Return splits from cross-validation method."""
    if cv_type == "stratified":
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=insight_config.RANDOM_SEED
        )
    elif cv_type == "logo":
        folds = LeaveOneGroupOut()
    elif cv_type == "group":
        folds = GroupKFold(n_splits=num_folds)
    else:
        print(
            "WARNING: Using KFold split likely to result in data leakage! Ensure no single person is in both the train and test set."
        )
        folds = KFold(
            n_splits=num_folds, shuffle=True, random_state=insight_config.RANDOM_SEED
        )
    return folds


def init_fold_metrics():
    fold_metrics = {
        "mcc": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    return fold_metrics


def get_fold_metrics(fold_metrics, y_true, y_preds) -> dict:
    fold_metrics["mcc"].append(matthews_corrcoef(y_true, y_preds))
    fold_metrics["accuracy"].append(accuracy_score(y_true, y_preds))
    fold_metrics["precision"].append(precision_score(y_true, y_preds, zero_division=0))
    fold_metrics["recall"].append(recall_score(y_true, y_preds, zero_division=0))
    fold_metrics["f1"].append(f1_score(y_true, y_preds, zero_division=0))
    return fold_metrics


def train(
    X: pd.DataFrame,
    feats,
    folds,
    target=None,
    groups=insight_config.CV_GROUP,
    models=None,
) -> Tuple[np.ndarray, dict, list, dict]:
    """Return trained models and predictions from cross-validation procedure.

    This is a basic function for exploratory model development and selection.
    Models are not retrained on full dataset after cross-validation;
    models from each fold are returned for further inspection.


    Parameters
    ----------
    X : pd.DataFrame
        training & validation dataset, including features, prediction target, and any group column to split by

    feats : pd.Index or list
        features to train on; DataFrame column names

    folds : iterator from sklearn's cv classes like GropuKFold
        returned from `train.get_cv_splits()` for cross-validation splits

    target : str
        DataFrame column name for the prediction target, e.g. "ptsd"

    models : list(str)
        names of models to train; if None, uses all models in MODEL_ZOO

    Returns
    -------
    oof_preds : np.ndarray
        out-of-fold predictions of shape (observations, 3 + number of models).
        First 3 columns are:
            - the CV-fold number
            - person's (factorized) unique id
            - ground truth label to check against the prediction
            These are useful in model scoring & evaluation.

    models : dict
        model names

    trained_models : list of length == len(models.values()) * insight_config.NUM_FOLDS
        fit models for each CV fold
    """
    if target is None:
        raise TypeError(
            "keyword argument `target` is required.\nSpecify the ground truth label as a column in the dataframe."
        )

    if models is None:
        models = model_zoo.BASIC_MODELS
    else:
        models = {m: model_zoo.BASIC_MODELS[m] for m in insight_config.MODELS}
    if "lgbm" in models.keys():
        models["lgbm"], lgb_params = model_zoo.init_lgb_params()
    if "catboost" in models.keys():
        models["catboost"], catboost_params = model_zoo.init_catboost_params()

    # Instantiate out-of-fold prediction placeholder
    EXTRA_COLUMNS = 3  # 3 extra columns for: tracking the fold, uids, and the labels for each CV fold
    oof_preds = np.full((X.shape[0], len(models) + EXTRA_COLUMNS), np.nan)
    # oof_scores = np.full((X.shape[0], 4), np.nan)

    # For logo variability tracking
    # if insight_config.CV_TYPE == "logo":
    fold_metrics = {k: init_fold_metrics() for k in models.keys()}

    # TODO: look into converting to:
    # cvscore = cross_val_score(clf, rocket_df.iloc[:, :9996].values, rocket_df['ptsd'].values, cv=logo, groups=groups, n_jobs=-1)
    # Gather split indexes for each fold & Train
    for n_fold, (train_idx, valid_idx) in enumerate(
        folds.split(X[feats], X[target], X[groups])
    ):
        print(f"\nn_fold: {n_fold} ------------------------------------")
        # print(f"n_fold: {n_fold}  valid_uids: {X['factorized_unique_id'].values[valid_idx]}")

        # Print labels for LOGO (too long)
        # train_labels = X[target].values[train_idx]
        # valid_labels = X[target].values[valid_idx]
        # print(f'train: {train_labels}')
        # print(f'valid: {valid_labels}')

        # Check for overlap people in training & validation
        training_uids = list(X["factorized_unique_id"].values[train_idx])
        validation_uids = list(X["factorized_unique_id"].values[valid_idx])
        overlap = list(set(validation_uids) & set(training_uids))
        if insight_config.VERBOSE:
            print(f"train/valid overlap: {len(overlap)}")
        if len(overlap) > 0:
            print("WARNING: SOMETHING WENT WRONG, TRAINING & VALIDATION UID OVERLAP")

        # Split the data: train/val & (X, y)
        X_train, y_train = X[feats].iloc[train_idx], X[target].iloc[train_idx]
        X_valid, y_valid = X[feats].iloc[valid_idx], X[target].iloc[valid_idx]

        if "lgbm" in models.keys():
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        # Set fold info in predictions array
        oof_preds[valid_idx, 0] = n_fold
        oof_preds[valid_idx, 1] = X["factorized_unique_id"].iloc[valid_idx]
        oof_preds[valid_idx, 2] = y_valid

        # Train models (Baselines & others)
        # TODO: upgrade to handle other models without tidy .fit() methods
        trained_models = []
        for i, (model_name, m) in enumerate(models.items()):
            # for i, m in enumerate(models.values()):
            # Train model
            # .fit() retrains a model, doesn't tune it from prevoius iteration
            # NOTE: when adding models, ensure warm_start=False
            if isinstance(m, lgb.LGBMClassifier):
                trained = lgb.train(  # trained is a lgb.Booster instance
                    params=lgb_params,
                    train_set=lgb_train,
                    valid_sets=(lgb_train, lgb_valid),
                    valid_names=("lgb_train", "lgb_valid"),
                    # callbacks=[wandb_callback()],
                    # categorical_feature=cccols,
                    # verbose_eval=300,
                )
            elif isinstance(m, CatBoostClassifier):
                trained = m.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    verbose=False,
                )
            else:
                trained = m.fit(X_train, y_train)
            trained_models.append(trained)

            # Predict on validation data
            # if hasattr(trained, "predict_proba") and callable(trained.predict_proba)
            column_idx = i + EXTRA_COLUMNS
            try:
                oof_preds[valid_idx, column_idx] = trained.predict_proba(X_valid)[:, 1]
            except:
                print(f"\nno predict proba for {trained}")
                try:
                    oof_preds[valid_idx, column_idx] = trained.predict(X_valid)
                except:
                    print(
                        f"\nadd something here no .predict_proba() or .predict() method for {trained}"
                    )
                    pass

            # Print brief score summaries for each model
            if insight_config.VERBOSE:
                # TODO: also consider printing class balance PER FOLD
                # Convert probas to preds
                y_preds = oof_preds[valid_idx, column_idx].copy()
                y_preds = insight_scorer.proba_to_pred(y_preds)

                # y_true = y_valid[valid_idx].copy()
                y_true = y_valid.copy()

                # Report Matthew's Corr Coefficient
                mcc = matthews_corrcoef(y_true, y_preds)
                if insight_config.CV_TYPE == "logo":
                    fold_metrics[model_name] = get_fold_metrics(
                        fold_metrics[model_name], y_true, y_preds
                    )
                # Show clean name for CatBoost, full details for others
                if isinstance(m, CatBoostClassifier):
                    print(f"\n--- CatBoostClassifier ---")
                else:
                    print(f"\n--- {m} ---")
                print(f"MCC:       {round(mcc, 2)}")
                # TODO: Add classification_report values for BOTH classes

    # Build predictions dataframe with useful extra columns
    cols = ["fold", "factorized_unique_id", "y_true"] + [
        model.__class__.__name__ for model in models.values()
    ]  # convert ugly model object to readable but informative model name
    predictions = pd.DataFrame(oof_preds, columns=cols)

    # Lazy future proofing:
    # Check for num columns mismatch (model dict vs util + model columns here)
    num_model_cols = predictions.shape[1] - EXTRA_COLUMNS
    assert num_model_cols == len(models)

    # Report df size
    if insight_config.VERBOSE:
        print(
            f"\nTrain returned {predictions.shape[0]} predictions from {num_model_cols} models."
        )
    return predictions, models, trained_models, fold_metrics


def factorize_categorical_column(df, col_name="unique_id") -> pd.DataFrame:
    """Add a new df column that converts the desired column to a numerical category.
    `pd.factorize()` alone encodes the object as an enumerated type or categorical
    variable.

    Example: run on `unique_id` column of dataframe for leave-one-person-out
        cross-validation, so that unique ids are ordered 0, 1, 2, ... instead
        of string literal `unique_id`s.
    """
    new_df = df.copy()
    new_col_name = "factorized_" + col_name
    new_df[new_col_name] = pd.factorize(new_df[col_name])[0]
    return new_df


def run_rocket(dataset, chunksize=512):
    print("Rocketing features...")
    dataset, rocket_columns = rockets.run_minirocket(
        dataset,
        timeseries_columns=insight_config.FEATS,
        target_col=insight_config.TARGET,
        chunksize=chunksize,
    )
    # save rocketed dataset
    print(f"Saving rocketed dataset, shape: {dataset.shape}")
    dataset.to_parquet("rocketed_dataset.parquet")
    return dataset, rocket_columns


def prep_data(dataset, rocket_columns=None):
    # TODO: legacy port, refactor later
    # Factorize uids for LOGO/GroupKFold cross-validation
    dataset = factorize_categorical_column(dataset, col_name="unique_id")
    # Wrap otuput for use in next functions
    prepped_data = dict({"dataset": dataset, "rocket_columns": rocket_columns})
    return prepped_data


def save_insight_config(args):
    """Save insight config constants as json"""
    cur_time = int(time.time())
    curdir = os.getcwd()
    filename = f"insight_config_{cur_time}.json"
    outpath = os.path.join(curdir, filename)

    config_dict = {
        "data_path": args.data_path,
        "rocket": args.rocket,
        "rkt_chunksize": args.rkt_chunksize,
        "VERBOSE": insight_config.VERBOSE,
        "DATASET": insight_config.DATASET,
        "INSPECT": insight_config.INSPECT,
        "INSPECT_SAVEPATH": insight_config.INSPECT_SAVEPATH,
        "TRAIN": insight_config.TRAIN,
        "HOLDOUT": insight_config.HOLDOUT,
        "SHOW_RESULTS": insight_config.SHOW_RESULTS,
        "OUTPATH": insight_config.OUTPATH,
        "COHORT": insight_config.COHORT,
        # "QUERY": insight_config.QUERY,
        # "DATABASE": insight_config.DATABASE,
        # "SITE_PREFIXES": insight_config.SITE_PREFIXES,
        # "TEST_SAMPLES": insight_config.TEST_SAMPLES,
        # "PTSD_LABELS_FILE": insight_config.PTSD_LABELS_FILE,
        "TARGET": insight_config.TARGET,
        "REMOVE_INTERMEDIATE_LABELS": insight_config.REMOVE_INTERMEDIATE_LABELS,
        "TASK": insight_config.TASK,  # TODO: update
        "SEG_FEATS": insight_config.SEG_FEATS,
        "GAZE_FEATS": insight_config.GAZE_FEATS,
        "CAT_FEATS": insight_config.CAT_FEATS,
        "USE_CAT_FEATS": insight_config.USE_CAT_FEATS,
        "FEATS": insight_config.FEATS,
        "UTIL_COLS": insight_config.UTIL_COLS,  # TODO: update
        "DEBLINK": insight_config.DEBLINK,
        "RUN_ROCKET": insight_config.RUN_ROCKET,
        "NUM_ROCKET_FEATS": insight_config.NUM_ROCKET_FEATS,
        "RANDOM_SEED": insight_config.RANDOM_SEED,
        "CV_TYPE": insight_config.CV_TYPE,
        "NUM_FOLDS": insight_config.NUM_FOLDS,
        "CV_GROUP": insight_config.CV_GROUP,
        "BALANCE_CLASSES": insight_config.BALANCE_CLASSES,
        "MODELS": insight_config.MODELS,
        "AUTOML": insight_config.AUTOML,
    }

    # Get all attributes from insight_config module
    config_attrs = [attr for attr in dir(insight_config) if not attr.startswith("_")]

    # Check for missing fields from insight_config
    config_dict_keys = set(config_dict.keys())
    # Remove non-config keys (args attributes)
    config_dict_keys.discard("data_path")
    config_dict_keys.discard("rocket")
    config_dict_keys.discard("rkt_chunksize")

    missing_attrs = set(config_attrs) - config_dict_keys
    if missing_attrs:
        print(
            f"Warning: config_dict is missing the following fields from insight_config: {sorted(missing_attrs)}"
        )

    with open(outpath, "w") as file:
        json.dump(config_dict, file)
    print(f"Saved training configuration {outpath}")
    return cur_time


##############################################################################
#### Higher level functions


def train_models(dataset_dict) -> Tuple[pd.DataFrame, dict, dict]:
    print("\n==============================================")
    print("Training Models...")

    if dataset_dict["rocket_columns"] is not None:
        feats = dataset_dict["rocket_columns"].copy()
    else:
        feats = insight_config.FEATS

    folds = get_cv_splits()

    # X, y conventions used for naming dataset (X) and labels (y) within this function
    X = dataset_dict["dataset"].copy()
    cv_predictions, model_names, trained_models, fold_metrics = train(
        X, feats, folds, target=insight_config.TARGET, models=insight_config.MODELS
    )

    return cv_predictions, model_names, trained_models, fold_metrics


def train_automl(
    dataset_dict, target=None, groups=insight_config.CV_GROUP
) -> Tuple[pd.DataFrame, dict, list, dict]:
    """Return trained AutoML models and predictions from cross-validation procedure.

    Similar to train() function but uses FLAML AutoML for model selection and
    hyperparameter optimization within each CV fold.

    Parameters
    ----------
    dataset_dict : dict
        Contains 'dataset' and 'rocket_columns' keys
    target : str
        DataFrame column name for the prediction target
    groups : str
        DataFrame column name for grouping in CV splits

    Returns
    -------
    predictions : pd.DataFrame
        out-of-fold predictions with same format as train()
    model_names : dict
        model names (will contain 'flaml_automl')
    trained_models : list
        fitted AutoML models for each CV fold
    fold_metrics : dict
        metrics for each fold
    """
    print("\n==============================================")
    print("Training AutoML Models...")

    if target is None:
        raise TypeError(
            "keyword argument `target` is required.\nSpecify the ground truth label as a column in the dataframe."
        )

    if dataset_dict["rocket_columns"] is not None:
        feats = dataset_dict["rocket_columns"].copy()
    else:
        feats = insight_config.FEATS

    folds = get_cv_splits()
    X = dataset_dict["dataset"].copy()

    # Setup for consistent output format with train()
    model_names = {"flaml_automl": "FLAML AutoML"}
    EXTRA_COLUMNS = 3  # fold, uid, y_true
    oof_preds = np.full((X.shape[0], len(model_names) + EXTRA_COLUMNS), np.nan)

    # Initialize fold metrics
    fold_metrics = {"flaml_automl": init_fold_metrics()}
    trained_models = []

    # Cross-validation loop
    for n_fold, (train_idx, valid_idx) in enumerate(
        folds.split(X[feats], X[target], X[groups])
    ):
        print(f"\nn_fold: {n_fold} ------------------------------------")

        # Check for overlap people in training & validation
        training_uids = list(X["factorized_unique_id"].values[train_idx])
        validation_uids = list(X["factorized_unique_id"].values[valid_idx])
        overlap = list(set(validation_uids) & set(training_uids))
        if insight_config.VERBOSE:
            print(f"train/valid overlap: {len(overlap)}")
        if len(overlap) > 0:
            print("WARNING: SOMETHING WENT WRONG, TRAINING & VALIDATION UID OVERLAP")

        # Split the data
        X_train, y_train = X[feats].iloc[train_idx], X[target].iloc[train_idx]
        X_valid, y_valid = X[feats].iloc[valid_idx], X[target].iloc[valid_idx]

        # Set fold info in predictions array
        oof_preds[valid_idx, 0] = n_fold
        oof_preds[valid_idx, 1] = X["factorized_unique_id"].iloc[valid_idx]
        oof_preds[valid_idx, 2] = y_valid

        # Train FLAML AutoML model
        print(f"Training FLAML AutoML for fold {n_fold}...")
        automl_model = model_zoo.run_flaml(X_train, y_train)
        trained_models.append(automl_model)

        # Get predictions
        try:
            y_proba = automl_model.predict_proba(X_valid)[:, 1]
            oof_preds[valid_idx, 3] = y_proba
        except Exception as e:
            print(f"Error getting predictions from FLAML: {e}")
            oof_preds[valid_idx, 3] = automl_model.predict(X_valid)

        # Calculate and store fold metrics
        if insight_config.VERBOSE:
            y_preds = insight_scorer.proba_to_pred(oof_preds[valid_idx, 3])
            y_true = y_valid.copy()

            mcc = matthews_corrcoef(y_true, y_preds)
            fold_metrics["flaml_automl"] = get_fold_metrics(
                fold_metrics["flaml_automl"], y_true, y_preds
            )
            print(f"\nFLAML AutoML")
            print(f"MCC:       {round(mcc, 2)}")

        # Feature importance for first fold only
        if n_fold == 0 and hasattr(automl_model, "feature_importances_"):
            feature_importance_pairs = list(
                zip(
                    automl_model.feature_names_in_,
                    automl_model.feature_importances_,
                )
            )
            top_10_features = sorted(
                feature_importance_pairs, key=lambda x: x[1], reverse=True
            )[:10]
            print("Top 10 most important features:")
            for feature_name, importance in top_10_features:
                print(f"{feature_name}: {importance}")

    # Build predictions dataframe
    cols = ["fold", "factorized_unique_id", "y_true"] + list(model_names.values())
    predictions = pd.DataFrame(oof_preds, columns=cols)

    # Generate learning curve plot
    try:
        (
            time_history,
            best_valid_loss_history,
            valid_loss_history,
            config_history,
            metric_history,
        ) = get_output_from_log(
            filename=".flaml-logs/flaml_experiment.log",
            time_budget=240,
        )

        plt.figure()
        plt.title("FLAML AutoML Learning Curve")
        plt.xlabel("Wall Clock Time (s)")
        plt.ylabel("Metric")
        plt.step(time_history, 1 - np.array(best_valid_loss_history), where="post")
        plt.savefig("flaml_learning_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved FLAML learning curve to flaml_learning_curve.png")
    except Exception as e:
        print(f"Could not generate learning curve: {e}")

    # Report dataset size
    if insight_config.VERBOSE:
        print(
            f"\nAutoML training returned {predictions.shape[0]} predictions from {len(model_names)} model(s)."
        )

    return predictions, model_names, trained_models, fold_metrics


def score_models(
    predictions,
    model_names,
    fold_metrics,
    dset_stats,
    insight_config,
    timestamp_id,
    fps,
):
    """Detailed scores & plots for model evaluation"""
    print("\nScoring Models...")

    # Prep ground truth, predicted probabilities (float 0-1), predicted classes (int)
    y_true = predictions["y_true"].values.astype("int")
    y_proba = predictions.iloc[:, 3:].values  # TODO: make this dynamic col ref
    y_pred = insight_scorer.proba_to_pred(y_proba)

    # Extract class balance numbers
    class_balance = dset_stats[2]

    # Print classification reports in the terminal for every model
    if insight_config.VERBOSE:
        insight_scorer.print_quick_reports(y_true, y_pred, model_names)

    # Plot model evaluation plots and save image
    if insight_config.SHOW_RESULTS:
        if len(model_names) == 1:
            insight_scorer.plot_results(
                y_true,
                y_proba,
                y_pred,
                model_names,
                dset_stats[2],  # Extract class_balance from dset_stats
                fold_metrics,
                timestamp_id,
                insight_config.COHORT,
                fps,
            )
        else:
            insight_scorer.plot_multi_results(
                y_true,
                y_proba,
                y_pred,
                model_names,
                dset_stats,
                fold_metrics,
                timestamp_id,
                insight_config.COHORT,
                fps,
            )


#######################################
#### Main FLow


def main(args):
    print("Loading data...")
    dataset = pd.read_parquet(args.data_path)
    # TODO: add multitarget support
    # for target in insight_config.TARGET
    dataset = dataset.dropna(subset=insight_config.TARGET)
    fps = get_fps_from_filename(args.data_path)
    if insight_config.VERBOSE:
        label_dist(dataset, col=insight_config.TARGET)

    # TODO: add filter for exclusion criteria

    # Filter tasks
    print(f"\nFiltering tasks to: {insight_config.TASK}...")
    dataset = dataset[dataset["task_id_renamed"].isin(insight_config.TASK)]
    if insight_config.VERBOSE:
        label_dist(dataset, col=insight_config.TARGET)

    # Filter intermediate labels
    if insight_config.REMOVE_INTERMEDIATE_LABELS:
        dataset = remove_intermediate_pcl(dataset)  # TODO: generalize beyond PCL
        if insight_config.VERBOSE:
            label_dist(dataset, col=insight_config.TARGET)

    # Filter cohorts
    print(f"\nFiltering cohorts to: {insight_config.COHORT}...")
    if isinstance(insight_config.COHORT, list):
        dataset = dataset[dataset["cohort"].isin(insight_config.COHORT)]
        if insight_config.VERBOSE:
            label_dist(dataset, col=insight_config.TARGET)
    elif isinstance(insight_config.COHORT, str):
        assert (
            insight_config.COHORT == "all"
        ), "`insight_config.COHORT` string not recognized"
        print("Using all cohorts")

    print(f"Num vids: {dataset['video_filename'].nunique()}")

    if args.rocket:
        dataset, rocket_columns = run_rocket(dataset, args.rkt_chunksize)

    prepped_data = prep_data(dataset, rocket_columns)
    dset_stats = dataset_stats(prepped_data["dataset"], target=insight_config.TARGET)

    # Initialize variables for scoring
    all_predictions = []
    all_model_names = {}
    all_fold_metrics = {}

    if insight_config.MODELS is not None:
        cv_predictions, model_names, trained_models, fold_metrics = train_models(
            prepped_data
        )
        all_predictions.append(cv_predictions)
        all_model_names.update(model_names)
        all_fold_metrics.update(fold_metrics)

    if insight_config.AUTOML:
        cv_predictions_aml, model_names_aml, trained_models_aml, fold_metrics_aml = (
            train_automl(prepped_data, target=insight_config.TARGET)
        )
        all_predictions.append(cv_predictions_aml)
        all_model_names.update(model_names_aml)
        all_fold_metrics.update(fold_metrics_aml)

    timestamp_id = save_insight_config(args)

    # Combine predictions if we have multiple sources, otherwise use single source
    if len(all_predictions) > 1:
        # Combine predictions from both regular models and AutoML
        base_cols = ["fold", "factorized_unique_id", "y_true"]
        combined_predictions = all_predictions[0][base_cols].copy()

        # Add model predictions from all sources
        for preds in all_predictions:
            model_cols = [col for col in preds.columns if col not in base_cols]
            combined_predictions = pd.concat(
                [combined_predictions, preds[model_cols]], axis=1
            )

        final_predictions = combined_predictions
    elif len(all_predictions) == 1:
        final_predictions = all_predictions[0]
    else:
        print(
            "No models were trained. Check insight_config.MODELS and insight_config.AUTOML settings."
        )
        return

    # Score all models together
    score_models(
        final_predictions,
        all_model_names,
        all_fold_metrics,
        dset_stats,
        insight_config,
        timestamp_id,
        fps,
    )
    # TODO: save_trained_models(trained, timestamp_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MiniROCKET feature extraction.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="path/to/your/timeseries/data.parquet",
        help="Path to the input timeseries data in parquet format.",
    )
    parser.add_argument(
        "--rkt-chunksize",
        type=int,
        default=512,
        help="Chunksize for MiniROCKET",
    )
    parser.add_argument(
        "--rocket",
        action="store_true",
        help="Use MiniROCKET for feature extraction.",
    )
    args = parser.parse_args()

    main(args)
