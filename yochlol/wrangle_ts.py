import os
import re
import warnings
from argparse import ArgumentParser
from typing import Tuple

import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import pandas as pd

plt.style.use("cyberpunk")

import awswrangler as wr
import torch
import torch.nn.functional as F
from tqdm import tqdm

# import fireducks.pandas as pd
# import dask.array as da
# import dask.dataframe as dd

FEATS = [
    "pupil_area_left",
    "pupil_area_right",
    "iris_area_left",
    "iris_area_right",
    "sclera_area_left",
    "sclera_area_right",
    "pupil_centers_left_x",
    "pupil_centers_left_y",
    "pupil_centers_right_x",
    "pupil_centers_right_y",
    "pirl",
    "pirr",
    "pog_x_coord",
    "pog_y_coord",
    "gaze_left_pitch",
    "gaze_left_yaw",
    "gaze_right_pitch",
    "gaze_right_yaw",
    "head_pitch",
    "head_yaw",
    "head_roll",
    # TODO: get interp cols from MC
]
UTIL_COLS = [
    "unique_id",
    "session_id",
    #'order_uuid',
    #'video_path',
    "video_filename",
    "cohort",
    "frametimestamps",
    "row_number",
    "block_number",  #
    #'event_order',
    #'event_name',
    #'event_start',
    #'event_end',
    "block_trial",  #
    "trial",  #
    "category",  #
    #'event_category',
    #'event_type',
    #'displayed_cal_dot_at_x',
    #'displayed_cal_dot_at_y',
    #'item',
    #'list_type',
    "phone_model",
    #'version_name',
    "task_id_renamed",
    "pcl_score",
    "ptsd_bin",
    "raps_binary_ptsd",
    # "raps_critB_bin",
    # "raps_critC_bin",
    # "raps_critD_bin",
    "raps_critE_bin",
    # "raps_critDISS_bin",
    "raps_score",
    "raps+pcl",
    "raps_weighted_pcl",
]
VERBOSE = True


def define_unique_groups(df, column="video_filename"):
    """Define grouby object and uniques, useful idiom for dataframe loop operations."""
    uniques = df[column].unique()
    grouped = df.groupby(column)
    return uniques, grouped


def label_dist(df, col="final_label"):
    # print(df.drop_duplicates('unique_id')[col].value_counts(dropna=False).sort_index())
    print(df.drop_duplicates("session_id")[col].value_counts(dropna=False).sort_index())


def dataset_stats(df, target="ptsd_bin"):
    """Return unique people, sessions, and class balance stats."""
    num_people = df["unique_id"].nunique()

    if "session_id" in df.columns.to_list():
        num_sessions = df["session_id"].nunique()
    else:
        num_sessions = np.nan

    # Return pd.Series of target labels by person ("unique_id")
    # TODO: also consider by "session_id"
    class_balance = df.drop_duplicates("unique_id")[target].value_counts().sort_index()
    return num_people, num_sessions, class_balance


def print_dataset_stats(num_people, num_sessions, class_balance):
    """Convenience to hide a bunch of printing in a function.
    Useful to know during exploration & data ingestion."""
    print("---------------------------------")
    print(f"Unique People: {num_people}")
    print(f"Num Sessions: {num_sessions}")
    print(f"Class balance:\n{class_balance}")
    print("---------------------------------")


def test_no_duplicates(df: pd.DataFrame, identifier: str, target_col: str = None):
    """Test df for common duplications/collisions after wrangling.

    NOTE: Intended use is on a dataframe (or df subset) corresponding
    to 1 unique id, 1 session, 1 video. Otherwise will hang.

    df : dataframe

    identifier : str
        video filename, unique_id, session_id, etc

    """
    if target_col is None:
        target_col = "ptsd_bin"
    assert df["unique_id"].nunique() == 1
    assert df["session_id"].nunique() == 1
    assert df["video_filename"].nunique() == 1
    # try:
    #     assert df["uploaded_at"].nunique() == 1
    # except AssertionError:
    #     unique_count = df["uploaded_at"].nunique()
    #     print(
    #         f'{unique_count} uniques in column "uploaded_at" in {identifier}. Expected 1, using first index'
    #     )

    # If this fails, it means the unique_id has 2 different PTSD labels
    try:
        assert df[target_col].nunique() == 1
    except:
        if df[target_col].isna().sum() > 0:
            print("NaNs detected in labels")
        else:
            raise Exception("dataframe has multiple labels")


def norm_and_interpolate(data, num_samples) -> np.ndarray:
    """Normalize and interpolate a signal.
    "Normalize" in this context means 0 mean, 1 standard dev.

    data : np.ndarray
        rows as observations, columns as features (1 col is a timeseries)

    num_samples : int
        desired length of output signal
    """
    # print(data.std(axis=0))
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    intrp = torch.from_numpy(data)
    data_interp = (
        F.interpolate(intrp.unsqueeze(0).unsqueeze(0), num_samples).squeeze().numpy()
    )
    return data_interp


def resample_integers(data, num_samples) -> np.ndarray:
    """Resample integer data without normalization.

    data : np.ndarray
        1D array of integer values to resample

    num_samples : int
        desired length of output signal
    """
    intrp = torch.from_numpy(data.astype(float))
    data_interp = (
        F.interpolate(intrp.unsqueeze(0).unsqueeze(0), num_samples).squeeze().numpy()
    )
    return data_interp


# def shortest_video_samples(timeseries_df: pd.DataFrame, task: str) -> int:
#     """Return the number of samples of the shortest video in your dataframe.

#     Assumes same task (PLR, affective img, cal, etc.), otherwise,
#     YOUR VIDS COULD BE CUT WAY TOO SHORT!

#     timeseries_df : dataframe with your timeseries
#     task : str like "plr" or "calibration"
#     """
#     task_samples = {  # 60fps
#         "calibration_1": 1650,  # 27.5 s
#         "calibration_2": 1650,
#         "plr": 480,  # 8 s
#         "face_pairs": 5460,  # 5.8-6.5 sec * 14 = (4827 to 5460)
#         "mckinnon": 1440,  # 24 s
#     }
#     num_samples = task_samples.get(task, None)

#     # Find shortest video
#     videos, grouped = define_unique_groups(timeseries_df, column="video_filename")
#     for vid in videos:
#         temp = grouped.get_group(vid).sort_values("row_number").reset_index(drop=True)
#         # uid = temp['unique_id'].unique()[0]

#         if temp.shape[0] < num_samples:
#             num_samples = temp.shape[0]

#     return num_samples


def get_60fps_nsamples(task: str) -> int:
    """Return the number of samples for a 60fps video task. Returns None if task not found."""
    TASK_SAMPLES = {
        "calibration_1": 1650,  # 27.5 s
        "calibration_2": 1650,
        "plr": 480,  # 8 s
        "face_pairs": 5460,  # 5.8-6.5 sec * 14 = (4827 to 5460)
        "mckinnon": 1440,  # 24 s
    }
    return TASK_SAMPLES.get(task, None)


def wrangle_timeseries(
    df,
    features=None,
    task=None,
    deblink=False,
    util_cols=UTIL_COLS,
    target_col="ptsd_bin",
    interp_factor=1.0,
) -> pd.DataFrame:
    """Prep timeseries data for loading into models.

    This includes trunacting time series, resampling, standardizing/normalizing, & deblinking.

    df : dataframe to process
    features : list of strings
        column names of features to include in model
    task : str or list of strings
        names of tasks to include e.g. PLR, calibration, ptsd images, etc.
    deblink : boolean
        deblink the timeseries or not

    """

    if task is None:
        task = "face_pairs"

    if features is None:
        features = FEATS

    # Filter df to specific task(s) and feature set
    if VERBOSE:
        print(f"Filtering to {task}")
    timeseries_df = df[df["task_id_renamed"] == task].reset_index(drop=True)
    if timeseries_df.empty:
        return pd.DataFrame()
    timeseries_df = timeseries_df[features + UTIL_COLS]

    # # Value to truncate samples to
    # shortest_timeseries = shortest_video_samples(timeseries_df, task)
    # if VERBOSE:
    #     print(f"Truncating {task} to {shortest_timeseries} samples")

    # Resample all to be same length, Normalize, & Repack into new DataFrame
    videos, grouped = define_unique_groups(timeseries_df, column="video_filename")
    df_collector = []

    n_skip = 0
    for vid in tqdm(videos):
        # for vid in videos[:20]:
        # temp = grouped.get_group(vid).sort_values("row_number").reset_index(drop=True)
        temp = (
            grouped.get_group(vid).sort_values("frametimestamps").reset_index(drop=True)
        )
        ts_nan = temp["frametimestamps"].isna().sum()
        assert ts_nan == 0  # or ignore if small number >0, remove is large >0

        # Test for common duplication issues in preprocessing
        # and Get values for this iteration subset
        try:
            assert target_col in temp.columns
        except:
            assert target_col in UTIL_COLS
        test_no_duplicates(temp, vid, target_col=target_col)
        uid = temp["unique_id"].iloc[0]
        sess = temp["session_id"].iloc[0]
        # upload_time = temp["uploaded_at"].iloc[0]
        label = temp[target_col].iloc[0]

        # Check if any feature has >= 50% NaN values
        feature_nans = temp[features].isna().sum()
        if (feature_nans >= 0.5 * temp.shape[0]).any():
            # print(f"Skipping video due to >= 50% NaN values in features")
            n_skip += 1
            continue
        # Convert nullable dtypes to float64 and handle NaN/NA values
        temp[features] = temp[features].astype("float64")
        temp[features] = temp[features].ffill().bfill()
        feature_nans = temp[features].isna().sum()
        # print(f"Total nans after processing: {feature_nans.sum()}")

        # Numericla util cols
        nutils = ["frametimestamps", "row_number"]
        assert temp[nutils].min().min() >= 0, "Negative values in util cols"
        temp[nutils] = temp[nutils].astype("int64")
        temp[nutils] = temp[nutils].ffill().bfill()
        nutil_nans = temp[nutils].isna().sum()
        if nutil_nans.sum() > 0:
            print(f"Total nans in util cols after processing: {nutil_nans.sum()}")

        try:
            # print(temp[features].dtypes)
            data = temp[features].values.astype("float")
        except:
            for col in temp[features].columns:
                if temp[col].isna().any():
                    idx = temp[temp[col].isna()]
                    print(col, idx)

        # Resample & Normalize timeseries
        # NEED TO RESAMPLE BEFORE IF IMPLEMENTING deblinking - check sampling rate for deblinker
        transformed_data = []
        for column_idx in range(data.shape[1]):
            column = data[:, column_idx].copy()
            n_samples = get_60fps_nsamples(task)
            # TODO: either feed smaller (trials) to interp to preserve event info
            # or use NN for event/category
            # NOTE: WARNING: Doesn't resample categorical or event data! Use with caution.
            tmp = norm_and_interpolate(
                column, int(n_samples * interp_factor)
            )  # TODO: play with interp_factor
            transformed_data.append(tmp)
        new_data = np.array(transformed_data).T

        # Resample integer utility columns without normalization
        transformed_utils = []
        nutils = temp[nutils].values
        for column_idx in range(nutils.shape[1]):
            column = nutils[:, column_idx].copy()
            n_samples = get_60fps_nsamples(task)
            tmp = resample_integers(column, int(n_samples * interp_factor))
            transformed_utils.append(tmp)
        new_tils = np.array(transformed_utils).T

        new_data = np.concatenate((new_tils, new_data), axis=1)

        # Autobots assemble
        ndf = pd.DataFrame(
            data=new_data, columns=["frametimestamps", "row_number"] + features
        )

        # New df columns
        ndf["unique_id"] = uid
        ndf["session_id"] = sess
        ndf["cohort"] = temp["cohort"].iloc[0]
        ndf["video_filename"] = vid
        ndf["task_id_renamed"] = task
        ndf["phone_model"] = temp["phone_model"].iloc[0]
        ndf["pcl_score"] = temp["pcl_score"].iloc[0]
        ndf[target_col] = label

        # TODO fixing...
        # # "unique_id",
        # # "session_id",
        # # "video_filename",
        # # "frametimestamps",
        # # "row_number",
        # "block_number",  #
        # "block_trial",  #
        # "trial",  #
        # "category",  #
        # # "phone_model",
        # #"task_id_renamed",
        # # "pcl_score",
        # # "ptsd_bin",

        # Collect eye features from each video
        df_collector.append(ndf)

    print(f"Skipped {n_skip}/{len(videos)} videos due to >= 50% NaNs")
    prepped_df = pd.concat(df_collector, axis=0).reset_index(drop=True)
    if VERBOSE:
        print(f"Timeseries wrangled from {features}")
        print(f"shape: {prepped_df.shape}")
        print(f"df columns:\n{prepped_df.columns.to_list()}")
    return prepped_df


def main(args):
    print("Loading data...")
    df2 = pd.read_parquet(args.data_path)
    df2["video_filename"] = df2["video_path"].apply(lambda x: os.path.basename(x))

    label_dist(df2, col=args.target)
    n_sessions = df2["session_id"].nunique()
    print(f"Sessions: {n_sessions}")

    ## Binary
    # Handle single task or multiple tasks
    if len(args.task) == 1:
        task = args.task[0]
        print(f"Wrangling...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dataset = wrangle_timeseries(
                df2,
                features=FEATS,
                task=task,
                deblink=args.deblink,
                target_col=args.target,
                interp_factor=args.interp_factor,
            )
    # Multiple Tasks
    else:
        dataset_collector = []
        for task in args.task:
            print(f"Wrangling...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                subset = wrangle_timeseries(
                    df2,
                    features=FEATS,
                    task=task,
                    deblink=args.deblink,
                    target_col=args.target,
                    interp_factor=args.interp_factor,
                )
                dataset_collector.append(subset)
        dataset = pd.concat(dataset_collector, axis=0)

    if args.save:
        sess = dataset["session_id"].nunique()
        current_time = int(pd.Timestamp.now().timestamp())
        new_fps = int(60 * args.interp_factor)  # assumes 60 fps base
        if sorted(args.task) == sorted(
            ["face_pairs", "calibration_1", "calibration_2", "plr", "mckinnon"]
        ):
            task_str = "all"
        else:
            task_str = task if len(args.task) == 1 else "_".join(args.task)
        fname = (
            f"wrangled_ts_{task_str}_{sess}sessions_{new_fps}fps_{current_time}.parquet"
        )
        print(f"Saving wrangled dataset to {fname}")
        savepath = os.path.join(args.savedir, fname)
        dataset.to_parquet(savepath)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--deblink",
        action="store_true",
        help="Remove & interpolate blinks",
    )
    parser.add_argument(
        "--data-path",
        default="MP_cohort_all_515sessions_1752504474_nocompliance.parquet",  # NOTE for fast developement
        help="Path to compliance statistics parquet file (default: compliance_stats.parquet)",
    )
    parser.add_argument(
        "--task",
        nargs="*",
        choices=["face_pairs", "calibration_1", "calibration_2", "plr", "mckinnon"],
        default=["face_pairs", "calibration_1", "calibration_2", "plr", "mckinnon"],
        help="Task(s) to process. Can be a single task or list of tasks. Valid options: face_pairs, calibration_1, calibration_2, plr, mckinnon. Default: all tasks",
    )
    parser.add_argument(
        "--interp-factor",
        type=float,
        default=1.0,
        help="Interpolation factor (<1 downsampling, >1 upsampling)",
    )
    parser.add_argument(
        "--savedir",
        default=os.getcwd(),
        help="Raw timeseries savedir",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the wrangled dataset to a parquet file",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["ptsd_bin", "raps_binary_ptsd", "raps_critE_bin"],
        default="ptsd_bin",
        help="Target column name to use as label",
    )
    # TODO: add options for interpolated, deblinked, smoothed cols from MC pipeline
    args = parser.parse_args()
    main(args)
