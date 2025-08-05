import os
from typing import Tuple

import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.style.use("cyberpunk")

import insight_config


def proba_to_pred(probas: np.ndarray, ge_threshold=0.5):
    """Convert predicted class probabilities (probas) between 0.-1. to predictions {0, 1}.

    probas : array of predictions between 0 and 1

    ge_threshold : rounding decision threshold
        Threshold is "greater than or equal to", meaning values equal to threshold will be rounded up to 1

    """
    predictions = probas.copy()
    predictions[predictions >= ge_threshold] = 1.0
    predictions[predictions < ge_threshold] = 0.0
    return predictions


def print_quick_reports(y_true, y_pred, model_names):
    """Print classification report for all models & folds

    y_true : np.ndarray
        ground truth class labels

    y_pred : np.ndarray
        class predictions (int, {0, 1})

    trained_models : list
    """
    for i, m in enumerate(model_names.values()):
        mcc = matthews_corrcoef(y_true, y_pred[:, i])
        # Show clean name for CatBoost, full details for others
        if isinstance(m, CatBoostClassifier):
            print(f"\n--- CatBoostClassifier ---")
        else:
            print(f"\n--- {m} ---")
        print(f"MCC:  {round(mcc, 2)}")
        print(classification_report(y_true, y_pred[:, i]))


def pr_curve_baseline(class_balance: pd.Series) -> float:
    """Calculate the baseline of the PR curve.

    Area under the PR curve is not as straightforward as the ROC AUC. Instead,
    the PR curve minimum in binary classification is directly related to the
    proportion of the positive class, see [0]

    References:
    [0]: Cao et al. (2020) The MCC-F1 curve: a performance evaluation technique
        for binary classification. https://arxiv.org/abs/2006.11278

    Parameters
    ----------
    class_balance : pd.Series
        class balance, 3rd return from `prep.dataset_stats()`, equivalent to dset_stats[2]

    Returns
    -------
    pr_baseline : float
        constant baseline value for PR-curve
    """
    n_neg = class_balance.sort_index().values[0]
    n_pos = class_balance.sort_index().values[1]
    return n_pos / (n_neg + n_pos)


def parse_binary_clf_report(clf_report: dict):
    """ "Return individual values from `sklearn.metrics.classification_report()` nested dict output.

    It's not pretty but it makes the `plot_results()` and `plot_multi_results()` a little more readable.
    """
    # Negative class PR metrics
    avg_prec_0 = round(clf_report["0"]["precision"], 2)
    avg_recall_0 = round(clf_report["0"]["recall"], 2)
    avg_f1_0 = round(clf_report["0"]["f1-score"], 2)
    support_0 = clf_report["0"]["support"]
    # Positive class PR metrics
    avg_prec_1 = round(clf_report["1"]["precision"], 2)
    avg_recall_1 = round(clf_report["1"]["recall"], 2)
    avg_f1_1 = round(clf_report["1"]["f1-score"], 2)
    support_1 = clf_report["1"]["support"]
    return (
        avg_prec_0,
        avg_recall_0,
        avg_f1_0,
        support_0,
        avg_prec_1,
        avg_recall_1,
        avg_f1_1,
        support_1,
    )


def plot_results(
    y_true,
    y_proba,
    y_pred,
    model_names,
    class_balance: pd.Series,
    fold_metrics,
    timestamp_id,
    cohort,
    fps,
):
    """Wrapper for plotting 1 model result."""

    # Initialize subplots
    WIDTH = 30
    NCOLS = 4
    height = 7 * len(model_names)
    fig, ax = plt.subplots(nrows=len(model_names), ncols=NCOLS, figsize=(WIDTH, height))
    # Set report font for last column
    mono = {"family": "monospace"}

    # Handle single model case - ax will be 1D array if nrows=1
    if len(model_names) == 1:
        ax = ax.reshape(1, -1)  # Reshape to 2D for consistent indexing

    # Since this is for 1 model, we don't need the loop - just process the single model
    m = list(model_names.keys())[0]  # Get the first (and only) model name

    # Calculate MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # Get classification report & unpack
    report = classification_report(y_true, y_pred, output_dict=True)
    (
        avg_prec_0,
        avg_recall_0,
        avg_f1_0,
        support_0,
        avg_prec_1,
        avg_recall_1,
        avg_f1_1,
        support_1,
    ) = parse_binary_clf_report(report)

    # 1 Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax[0, 0])
    ax[0, 0].grid(False)
    ax[0, 0].set_title("Confusion Matrix")

    # 1 ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    rocauc = roc_auc_score(y_true, y_proba)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=rocauc, estimator_name=m).plot(
        ax=ax[0, 1]
    )
    ax[0, 1].plot([0, 1], [0, 1], "--", lw=1)
    ax[0, 1].set_title(
        f"Model: {model_names[m]}\nMCC: {round(mcc, 2)}", fontsize=16, weight="bold"
    )
    mplcyberpunk.add_underglow(ax[0, 1])

    # 1 PR curve
    pr_baseline = pr_curve_baseline(class_balance)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    display = PrecisionRecallDisplay(precision=precision, recall=recall).plot(
        ax=ax[0, 2]
    )
    ax[0, 2].plot([0, 1], [pr_baseline, pr_baseline], "--", lw=1)
    ax[0, 2].set_title(f"Precision-Recall Curve")
    ax[0, 2].set_ylim([0, 1])

    # Add classification report info to subplots
    ax[0, 3].grid(False)
    ax[0, 3].set_xticklabels([])
    ax[0, 3].set_yticklabels([])
    # Format text
    heading_str = "\t    prec.   recall  f1-score\tsupport".expandtabs()
    class0_str = f"class 0:\t{avg_prec_0}\t{avg_recall_0}\t{avg_f1_0}        {support_0}".expandtabs()
    class1_str = f"class 1:\t{avg_prec_1}\t{avg_recall_1}\t{avg_f1_1}        {support_1}".expandtabs()
    ax[0, 3].text(0.1, 1.0, heading_str, fontdict=mono, fontsize=16)
    ax[0, 3].text(0.0, 0.9, class0_str, fontdict=mono, fontsize=16)
    ax[0, 3].text(0.0, 0.8, class1_str, fontdict=mono, fontsize=16)

    # Add dataset statistics to the bottom right subplot
    dset_stats_str = f"People per class:\n{class_balance.to_string()}"
    ax[0, 3].text(0.0, 0.6, dset_stats_str, fontdict=mono, fontsize=16, va="top")

    cohort_str = f"Cohort: {cohort}"
    ax[0, 3].text(0.6, 0.6, cohort_str, fontdict=mono, fontsize=16, va="top")

    # Note: We don't have total_people and total_sessions from dset_stats in single model version
    # so we'll calculate total people from class_balance
    total_people = class_balance.sum()
    total_people_str = f"Total people: {total_people}"
    ax[0, 3].text(0.0, 0.3, total_people_str, fontdict=mono, fontsize=16, va="top")

    ax[0, 3].text(0.0, 0.15, f"fps: {fps}", fontdict=mono, fontsize=16, va="top")

    if len(str(insight_config.TASK)) > 50:
        task_list_formatted = ",\n  ".join(
            [f'"{task}"' for task in insight_config.TASK]
        )
        task_str = f"Tasks: [\n  {task_list_formatted}\n]"
    else:
        task_str = f"Tasks: {insight_config.TASK}"
    ax[0, 3].text(0.6, 0.3, task_str, fontdict=mono, fontsize=16, va="top")

    # Save Figures a img
    curdir = os.getcwd()
    filename = f"insight_scores_{timestamp_id}.png"
    outpath = os.path.join(curdir, filename)
    plt.savefig(outpath)
    print(f"Saved results report at {outpath}")


def plot_multi_results(
    y_true,
    y_proba,
    y_pred,
    model_names,
    dset_stats,
    fold_metrics,
    timestamp_id,
    cohort,
    fps,
):
    """Plot results for multiple models in 1 large subplot."""

    # Initialize subplots
    WIDTH = 30
    NCOLS = 4
    height = 7 * len(model_names)
    fig, ax = plt.subplots(nrows=len(model_names), ncols=NCOLS, figsize=(WIDTH, height))
    # Set report font for last column
    mono = {"family": "monospace"}

    # Make the plots
    for i, m in enumerate(model_names):
        # mcc = matthews_corrcoef(y_true, y_pred)
        # TODO: use this in score.py
        mcc = matthews_corrcoef(y_true, y_pred[:, i])

        # Get classification report & unpack
        report = classification_report(y_true, y_pred[:, i], output_dict=True)
        (
            avg_prec_0,
            avg_recall_0,
            avg_f1_0,
            support_0,
            avg_prec_1,
            avg_recall_1,
            avg_f1_1,
            support_1,
        ) = parse_binary_clf_report(report)

        # 1 Confusion matrix per model
        cm = confusion_matrix(y_true, y_pred[:, i])
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax[i, 0])
        ax[i, 0].grid(False)
        ax[i, 0].set_title("Confusion Matrix")

        # TODO: remove this debugging for roc jagged test
        # print("y_proba shape")
        # print(y_proba.shape)
        # print("y_proba [:, i] shape")
        # print(y_proba[:, i].shape)

        # 1 ROC curve per model
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, i])
        rocauc = roc_auc_score(y_true, y_proba[:, i])
        # Show clean name for CatBoost, full details for others
        model_obj = model_names[m]
        if isinstance(model_obj, CatBoostClassifier):
            model_display_name = "CatBoostClassifier"
        else:
            model_display_name = str(model_obj)
            
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=rocauc, estimator_name=model_display_name
        ).plot(ax=ax[i, 1])
        ax[i, 1].plot([0, 1], [0, 1], "--", lw=1)
        ax[i, 1].set_title(
            f"Model: {model_display_name}\nMCC: {round(mcc, 2)}", fontsize=16, weight="bold"
        )
        mplcyberpunk.add_underglow(ax[i, 1])

        # 1 PR curve per model
        class_balance = dset_stats[2]
        pr_baseline = pr_curve_baseline(class_balance)
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, i])
        display = PrecisionRecallDisplay(precision=precision, recall=recall).plot(
            ax=ax[i, 2]
        )
        ax[i, 2].plot([0, 1], [pr_baseline, pr_baseline], "--", lw=1)
        ax[i, 2].set_title(f"Precision-Recall Curve")
        ax[i, 2].set_ylim([0, 1])

        # Add classification report info to subplots
        ax[i, 3].grid(False)
        ax[i, 3].set_xticklabels([])
        ax[i, 3].set_yticklabels([])
        # Format text
        heading_str = "\t    prec.   recall  f1-score\tsupport".expandtabs()
        class0_str = f"class 0:\t{avg_prec_0}\t{avg_recall_0}\t{avg_f1_0}        {support_0}".expandtabs()
        class1_str = f"class 1:\t{avg_prec_1}\t{avg_recall_1}\t{avg_f1_1}        {support_1}".expandtabs()
        ax[i, 3].text(0.1, 1.0, heading_str, fontdict=mono, fontsize=16)
        ax[i, 3].text(0.0, 0.9, class0_str, fontdict=mono, fontsize=16)
        ax[i, 3].text(0.0, 0.8, class1_str, fontdict=mono, fontsize=16)

        # Add dataset statistics to the bottom right subplot
        if i == len(model_names) - 1:  # Last model (bottom row)
            dset_stats_str = f"People per class:\n{dset_stats[2].to_string()}"
            ax[i, 3].text(
                0.0, 0.6, dset_stats_str, fontdict=mono, fontsize=16, va="top"
            )

            cohort_str = f"Cohort: {cohort}"
            ax[i, 3].text(0.6, 0.6, cohort_str, fontdict=mono, fontsize=16, va="top")

            total_people_str = f"Total people: {dset_stats[0]}"
            ax[i, 3].text(
                0.0, 0.3, total_people_str, fontdict=mono, fontsize=16, va="top"
            )

            total_sessions_str = f"Total sessions: {dset_stats[1]}"
            ax[i, 3].text(
                0.0, 0.25, total_sessions_str, fontdict=mono, fontsize=16, va="top"
            )

            ax[i, 3].text(
                0.0, 0.15, f"fps: {fps}", fontdict=mono, fontsize=16, va="top"
            )

            if len(str(insight_config.TASK)) > 50:
                task_list_formatted = ",\n  ".join(
                    [f'"{task}"' for task in insight_config.TASK]
                )
                task_str = f"Tasks: [\n  {task_list_formatted}\n]"
            else:
                task_str = f"Tasks: {insight_config.TASK}"
            ax[i, 3].text(0.6, 0.3, task_str, fontdict=mono, fontsize=16, va="top")

    # Save Figures a img
    curdir = os.getcwd()
    filename = f"insight_scores_{timestamp_id}.png"
    outpath = os.path.join(curdir, filename)
    plt.savefig(outpath)
    print(f"Saved results report at {outpath}")
