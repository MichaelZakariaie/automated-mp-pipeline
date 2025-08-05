import os
import re
import time
import warnings
from argparse import ArgumentParser

import awswrangler as wr
import boto3

# import fireducks.pandas as pd
import numpy as np
import pandas as pd
from botocore.config import Config
from tqdm import tqdm

# import dask.array as da
# import dask.dataframe as dd


def define_unique_groups(df, column="video_filename"):
    """Define grouby object and uniques, useful idiom for dataframe loop operations."""
    uniques = df[column].unique()
    grouped = df.groupby(column)
    return uniques, grouped


def label_dist(df, col="final_label"):
    # print(df.drop_duplicates('unique_id')[col].value_counts(dropna=False).sort_index())
    print(df.drop_duplicates("session_id")[col].value_counts(dropna=False).sort_index())


def filter_cohorts(df, cohort) -> pd.DataFrame:
    """Filter dataframe by cohort number(s)."""
    if isinstance(cohort, int):
        return df[df["cohort"] == cohort].reset_index(drop=True)
    elif isinstance(cohort, tuple) or isinstance(cohort, list):
        return df[df["cohort"].isin(cohort)].reset_index(drop=True)
    else:
        raise ValueError("Cohort must be an int or a list/tuple of ints.")


def get_task_names_map():
    """Return mapping use to unify various names for each task"""
    return {
        # "plr_view": "plr",
        # "PLR": "plr",
        "aiv": "affective_image_set",
        "ptsd_image_set": "affective_image_set",
    }


def fix_task_names(df: pd.DataFrame) -> pd.DataFrame:
    """Returns df with new task_id_renamed column with old intact.

    Effectively, removes block numbers and other inconsistent naming practices.
    """

    task_dict = get_task_names_map()
    new_df = df.copy()

    # Must create new column each renaming step.
    # Re-running on same column overwrites ALL values, effectively only running that line
    new_df["renamed_0"] = new_df["task_id"].str.replace(  # plr_x -> plr
        r".*plr.*", "plr", flags=re.IGNORECASE, regex=True
    )
    new_df["renamed_1"] = new_df["renamed_0"].replace(task_dict)  # tasks in task dict
    new_df["renamed_2"] = new_df["renamed_1"].str.replace(  # aiv block removal
        r".*affective.*", "affective_image_set", regex=True
    )
    new_df["renamed_3"] = new_df["renamed_2"].str.replace(
        r".*face_pairs.*", "face_pairs", regex=True
    )
    new_df["renamed_4"] = new_df["renamed_3"].str.replace(
        r".*mckinnon.*", "mckinnon", regex=True
    )
    new_df["renamed_5"] = new_df["renamed_4"].str.replace(
        r".*list_learning.*", "list_learning", regex=True
    )

    new_df["task_id_renamed"] = new_df["renamed_5"].str.replace(
        r".*attention.*", "attention_bias", flags=re.IGNORECASE, regex=True
    )
    # TODO: check other dot stuff like events

    del new_df["renamed_0"]
    del new_df["renamed_1"]
    del new_df["renamed_2"]
    del new_df["renamed_3"]
    del new_df["renamed_4"]
    del new_df["renamed_5"]

    # Test for misses (additional tasks beyond this will result in False)
    REFERENCE = [
        "plr",
        "calibration_1",
        "calibration_2",
        # "attention_bias",
        "face_pairs",
        # "affective_image_set",
        "mckinnon",
        "list_learning",
    ]
    tasks = new_df["task_id_renamed"].value_counts().index.to_list()
    assert all(t in REFERENCE for t in tasks)
    return new_df


def no_nans(series: pd.Series):
    return (series.isna()).sum() == 0


def convert_pog2(device, x_pt_arr, y_pt_arr):
    """convert phone coordinate to POG coordinate, Research Team verified"""
    ppi = 460
    if "Pro Max" in device:
        screen_cm, x_offset, y_offset = 7.12, 2.9, 0.51
    elif "Pro" in device:
        screen_cm, x_offset, y_offset = 6.5, 2.5, 0.47
    elif "Plus" in device:
        screen_cm, x_offset, y_offset = 6.9, 2.6, 0.49
    elif any(f"iPhone {i}" in device for i in range(12, 17)):
        screen_cm, x_offset, y_offset = 6.45, 2.35, 0.5
    else:
        screen_cm, x_offset, y_offset = 6.5, 2.5, 0.5
    x_cm = ((x_pt_arr * 3) / ppi) * 2.54
    y_cm = ((y_pt_arr * 3) / ppi) * 2.54
    x_true = (screen_cm - x_cm) - x_offset
    y_true = y_cm - y_offset

    if np.all(np.isnan(x_true)):
        x_true2 = x_true  # keep it all NaNs
    else:
        x_range = np.nanmax(x_true) - np.nanmin(x_true)
        x_true2 = (-1 * x_true) + x_range

    return [x_true2, y_true]


def compute_POG_cal_error(df_cal, phone_model):
    """POG error in Research Team format"""

    df_cal["displayed_cal_dot_at_x"] = df_cal["displayed_cal_dot_at_x"].astype("float")
    df_cal["displayed_cal_dot_at_y"] = df_cal["displayed_cal_dot_at_y"].astype("float")

    all_cal = pd.DataFrame()
    for ind_cal in df_cal["task_id"].unique():

        cal1 = df_cal[df_cal["task_id"] == ind_cal]

        true_x_cal = cal1["displayed_cal_dot_at_x"].values
        true_y_cal = cal1["displayed_cal_dot_at_y"].values

        rescale_tru_x, rescale_tru_y = convert_pog2(phone_model, true_x_cal, true_y_cal)

        cal1["displayed_cal_dot_at_x_corrected"] = rescale_tru_x
        cal1["displayed_cal_dot_at_y_corrected"] = rescale_tru_y

        cal1_err = pd.DataFrame()
        pog_error_x2 = []
        pog_error_y2 = []
        pog_std_x2 = []
        pog_std_y2 = []

        for ind_dot in cal1["event_order"].unique():
            # cv_dot = pd.DataFrame()
            # try:
            if ind_dot == 12:
                pass
            else:

                cv_dot = cal1[cal1["event_order"] == ind_dot]
                # print(cv_dot.shape)
                tru_x = cv_dot["displayed_cal_dot_at_x_corrected"].unique()
                tru_y = cv_dot["displayed_cal_dot_at_y_corrected"].unique()

                pog_x_mean = np.nanmean(
                    cv_dot["pog_x_coord"][30:]
                )  # giving 500 ms of saccade time to be generous
                pog_y_mean = np.nanmean(cv_dot["pog_y_coord"][30:])

                pog_x_std = np.nanstd(cv_dot["pog_x_coord"][30:])
                pog_y_std = np.nanstd(cv_dot["pog_y_coord"][30:])

                pog_error_x = abs(tru_x - pog_x_mean)
                pog_error_y = abs(tru_y - pog_y_mean)

                if (pog_error_x.item() >= 2) | (pog_error_y.item() >= 2):
                    cv_dot["per_dot_threshold_both2"] = "fail"  #
                elif (pog_error_x.item() < 2) & (pog_error_y.item() < 2):
                    cv_dot["per_dot_threshold_both2"] = "pass"
                else:
                    cv_dot["per_dot_threshold_both2"] = ""

                if (pog_error_x.item() >= 3) | (pog_error_y.item() >= 3):
                    cv_dot["per_dot_threshold_both3"] = "fail"
                elif (pog_error_x.item() < 3) & (pog_error_y.item() < 3):
                    cv_dot["per_dot_threshold_both3"] = "pass"
                else:
                    cv_dot["per_dot_threshold_both3"] = ""

                if pog_error_y.item() >= 3:
                    cv_dot["per_dot_threshold_y3"] = "fail"
                elif pog_error_y.item() < 3:
                    cv_dot["per_dot_threshold_y3"] = "pass"
                else:
                    cv_dot["per_dot_threshold_y3"] = ""

                if pog_error_y.item() >= 2:
                    cv_dot["per_dot_threshold_y2"] = "fail"
                elif pog_error_y.item() < 2:
                    cv_dot["per_dot_threshold_y2"] = "pass"
                else:
                    cv_dot["per_dot_threshold_y2"] = ""

                pog_error_x2.append(pog_error_x.item())
                pog_error_y2.append(pog_error_y.item())
                pog_std_x2.append(pog_x_std.item())
                pog_std_y2.append(pog_y_std.item())

                cal1_err = pd.concat([cal1_err, cv_dot])

                VALS_COL = [
                    "per_dot_threshold_both2",
                    "per_dot_threshold_both3",
                    "per_dot_threshold_y3",
                    "per_dot_threshold_y2",
                ]

                vals_dict = {}

                for var_col in VALS_COL:
                    vals_dict[var_col] = (
                        cal1_err.groupby("event_order")[var_col]
                        .unique()
                        .apply(lambda x: x.item() if len(x) == 1 else np.nan)
                    )

                vals = pd.DataFrame(vals_dict).reset_index()
                vals2 = vals

                vals2["session_id"] = df_cal["session_id"].iloc[0]
                vals2["task_id"] = ind_cal
                vals2["video_path"] = df_cal["video_path"].iloc[0]

                vals2["pog_error_x"] = pog_error_x2
                vals2["pog_error_y"] = pog_error_y2

                vals2["pog_x_std"] = pog_std_x2
                vals2["pog_y_std"] = pog_std_y2

        all_cal = pd.concat([all_cal, vals2])
    return all_cal


def remove_intermediate_pcl(
    dx: pd.DataFrame, ptsd_threshold=33, buffer=8
) -> pd.DataFrame:
    """Remove people who scored near the PTSD threshold on PCL (Dr. Rothbaum).
    Might improve models.

    dx : pcl dataframe from a query
    """
    print("Removing intermediate PCL scores...")
    # e.g. ptsd_threshold = 33, buffer = 8 => PCL (25, 41)
    dx = dx[
        ~dx["pcl_score"].between(
            ptsd_threshold - buffer, ptsd_threshold + buffer, inclusive="both"
        )
    ]
    print(dx["ptsd_bin"].value_counts(dropna=False).sort_index())
    return dx


def save_df(df, args, save_dir):
    cohort = args.filter_cohorts if args.filter_cohorts is not None else "all"
    num_sessions = df["session_id"].nunique()
    current_time = int(pd.Timestamp.now().timestamp())
    if args.bypass_comp:
        fname = f"MP_cohort_{cohort}_{num_sessions}sessions_{current_time}_nocompliance.parquet"
    else:
        fname = f"MP_cohort_{cohort}_{num_sessions}sessions_{current_time}.parquet"
    savepath = os.path.join(save_dir, fname)
    print(f"Saving to {savepath}")
    df.to_parquet(savepath)


def main(args):
    print("Querying glue tables, this will take ~15 mins...")

    # Configure boto3 client with extended timeouts and retries
    config = Config(
        read_timeout=900,  # 15 minutes
        connect_timeout=60,
        retries={"max_attempts": 10, "mode": "adaptive"},
    )

    # Set the boto3 session configuration
    boto3.setup_default_session()
    session = boto3.Session()

    # Configure awswrangler to use our session
    wr.config.s3_endpoint_url = None
    wr.config.s3_additional_kwargs = {"Config": config}

    # Qualtrics
    print("Getting Qualtrics...")
    QUERY = "select * from data_quality.qualtrics_surveys_unified"
    DATABASE = "data_quality"
    df = wr.athena.read_sql_query(
        QUERY,
        database=DATABASE,
        ctas_approach=True,
        chunksize=True,
        boto3_session=session,
        use_threads=True,
    )
    # concat the generator of dataframes returned by chunksize=True
    dfq = pd.concat(df, axis=0)

    # PCL processed
    print("Getting PCL labels...")
    QUERY = "select * from data_quality.mp_pcl_scores"
    DATABASE = "data_quality"
    df = wr.athena.read_sql_query(
        QUERY,
        database=DATABASE,
        ctas_approach=True,
        chunksize=True,
        boto3_session=session,
        use_threads=True,
    )
    dx = pd.concat(df, axis=0)
    # RAPS
    print("Getting RAPS labels...")
    QUERY = "select * from data_quality.mp_raps_scores"
    DATABASE = "data_quality"
    df = wr.athena.read_sql_query(
        QUERY,
        database=DATABASE,
        ctas_approach=True,
        chunksize=True,
        boto3_session=session,
        use_threads=True,
    )
    dr = pd.concat(df, axis=0)
    dr["raps_binary_ptsd"] = dr["ptsd"].map({"Negative": 0, "Positive": 1})
    dr["raps_critE_bin"] = dr["arousal and reactivity symptoms criteria met"].map(
        {"no": 0, "yes": 1}
    )

    # Merge PCL & RAPS
    dx = dx.merge(
        dr[["session_id", "raps_score", "raps_binary_ptsd", "raps_critE_bin"]],
        on="session_id",
        how="left",
    )
    if args.filter_cohorts is not None:
        # TODO: dx['cohort'] = dx['cohort'].astype(int)
        dx = filter_cohorts(dx, cohort=args.filter_cohorts)
    # TODO: grab PHQ and GAD scores as well
    # data_quality.mp_phq_scores # depression
    # data_quality.mp_gad_scores # anxiety
    # data_quality.mp_raps_surveys

    # Combine RAPS and PCL labels
    dx["raps+pcl"] = dx["raps_score"] + dx["pcl_score"]
    dx["raps_weighted_pcl"] = dx["raps_score"] * dx["pcl_score"]
    print(dx.shape)

    # Quality stuff
    print("Getting video quality info...")
    QUERY = "select session_id, problem from data_quality.master_query_session_completion_check where problem = 'none' and study_location = 'Messy Prototyping'"
    DATABASE = "data_quality"
    df = wr.athena.read_sql_query(
        QUERY,
        database=DATABASE,
        ctas_approach=True,
        chunksize=True,
        boto3_session=session,
        use_threads=True,
    )
    df_quality = pd.concat(df, axis=0)

    # Overall session filtering before CV & E makes query faster
    # Use intersection of cohort-filtered dx and quality-filtered sessions
    sessions_to_use = list(
        set(dx["session_id"].unique()) & set(df_quality["session_id"].unique())
    )

    # Hermit CV & Events - with retry logic for timeouts
    print(f"Getting CV and Event data from {len(sessions_to_use)} sessions...")
    # Filter by sessions that passed both cohort and quality filters
    sessions_filter = "', '".join(sessions_to_use)
    QUERY = f"""
        select * from data_quality.messy_prototyping_app_session_details
        where session_id in ('{sessions_filter}')
        """
    DATABASE = "data_quality"
    max_retries = 3
    retry_delay = 30  # seconds
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} for CV and Event data...")
            df = wr.athena.read_sql_query(
                QUERY,
                database=DATABASE,
                ctas_approach=True,
                chunksize=True,
                boto3_session=session,
                use_threads=True,
            )
            print("    Query succeeded, concatenating generator...")
            dfcve = pd.concat(df, axis=0)
            print("Successfully retrieved CV and Event data")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # exponential backoff
            else:
                print("All retry attempts failed")
                raise
    dfcve["unique_id"] = dfcve["session_id"].apply(lambda x: x.split("_")[0])

    # Check that qualtrics and CV/events are equal size
    # try:
    #     assert (
    #         dfq["recipient_id"].nunique() == dfq["session_id"].nunique()
    #     ), "Unequal session & recipient id"
    # except AssertionError as e:
    #     if dfq["recipient_id"].nunique() < dfq["session_id"].nunique():
    #         print("More sessions than recipient ids. Repeat sessions?")
    assert dfcve["session_id"].nunique() == dfcve["session_id"].nunique()
    assert dx["session_id"].nunique() == dx["session_id"].nunique()

    ##### Merge/intersections
    dfq = dfq[dfq["session_id"].isin(sessions_to_use)].reset_index(drop=True)
    dfcve = dfcve[dfcve["session_id"].isin(sessions_to_use)].reset_index(drop=True)
    dx = dx[dx["session_id"].isin(sessions_to_use)].reset_index(drop=True)
    # Filter intersection of all 3 and check shapes
    q, c, x = (
        dfq["session_id"].unique(),
        dfcve["session_id"].unique(),
        dx["session_id"].unique(),
    )
    q, c, x = set(q), set(c), set(x)
    intersection = c & x
    print(f"CV & PCL intersection: {len(intersection)}")
    final_session_list = list(q & intersection)
    print(f"+ Qualtrics: {len(final_session_list)}")

    dfq = dfq[dfq["session_id"].isin(final_session_list)].reset_index(drop=True)
    dfcve = dfcve[dfcve["session_id"].isin(final_session_list)].reset_index(drop=True)
    dx = dx[dx["session_id"].isin(final_session_list)].reset_index(drop=True)

    assert (
        dfq["session_id"].nunique()
        == dfcve["session_id"].nunique()
        == dx["session_id"].nunique()
    )

    # Binary labels
    dx["ptsd_bin"] = dx["ptsd"].map({"Negative": 0, "Positive": 1})
    print(dx["ptsd_bin"].value_counts(dropna=False).sort_index())

    # Split PCL
    if args.split_pcl:
        dx = remove_intermediate_pcl(dx)

    #####################
    # Drop List Learning / .m4a / audio from dataframe, it has no CV/timeseries
    idx_LL = dfcve["task_id"].str.contains("list_learning")
    dfcve = dfcve[~idx_LL]
    # TODO: replace later with tabular data

    # Order
    dfcve = dfcve.sort_values(by=["session_id", "frametimestamps"]).reset_index(
        drop=True
    )

    # Grab phone model info
    print("Getting phone model info...")
    EXCLUDE_PHONES = ["?unrecognized?", "iPhone 13 Mini"]
    QUERY = f"""
            SELECT DISTINCT *
            FROM data_quality.master_query_session_completion_check
            WHERE study_location = 'Messy Prototyping'
            AND problem = 'none'
            """
    # db = wr.athena.read_sql_query(QUERY, database="data_quality", boto3_session=session)
    db = wr.athena.read_sql_query(
        QUERY, database="data_quality", boto3_session=session, use_threads=True
    )
    db = db[db["session_id"].isin(final_session_list)][["session_id", "phone_model"]]
    db = db[~db["phone_model"].isin(EXCLUDE_PHONES)]
    db = db[~db["phone_model"].str.contains("Mini")]
    # merge phone type with event/cv df
    dfcve = db.merge(dfcve, how="inner", on="session_id").reset_index(drop=True)

    # Rename
    print(f"dfcve2 shape {dfcve.shape}")
    print("Fixing task names...")
    dfcve2 = fix_task_names(dfcve)
    print(f"dfcve2 shape {dfcve2.shape}")
    print(f"dx shape {dx.shape}")

    dfcve2 = dfcve2.dropna(subset="frametimestamps", axis=0).reset_index(drop=True)

    # nan check
    assert no_nans(dfcve2["session_id"])
    # assert no_nans(dfcve2['video_path'])
    assert no_nans(dfcve2["row_number"])  # bypass
    assert no_nans(dfcve2["frametimestamps"])

    # Merge labels with rest (CV etc)
    print("Merging...")
    labels = dx[
        [
            "session_id",
            "cohort",
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
    ]
    print(f"dfcve2 shape {dfcve2.shape}")
    print(f"labels shape {labels.shape}")
    df = labels.merge(dfcve2, how="inner", on="session_id")
    print(df.shape, df["session_id"].nunique())
    label_dist(df, col="ptsd_bin")
    # remove extra 3000ms of the get ready app segment
    df = df[~df["event_name"].str.contains("get ready")].reset_index(drop=True)

    # Add POG calibration error columns (Research/Veronica)
    print("Getting Research POG err...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # get cal tasks
        cal_df2 = df[df["task_id_renamed"].isin(["calibration_1", "calibration_2"])]
        # TODO : empty check
        all_cal_vids, df2g = define_unique_groups(cal_df2, column="video_path")
        # all_sess_cal = pd.DataFrame()
        df_collector = []
        for vid in tqdm(all_cal_vids):
            all_cal = pd.DataFrame()
            df_cal2 = df2g.get_group(vid)
            if df_cal2.empty:
                continue
            df_cal = df_cal2.dropna(subset=["event_name"])
            COLS = ["pog_y_coord", "pog_x_coord"]
            df_cal[COLS] = df_cal[COLS].replace(["0", 0], np.nan)
            phone_model = cal_df2[cal_df2["video_path"] == vid]["phone_model"].unique()
            if len(phone_model) > 1:
                print(f"WARNING: multiple phone models reported in {vid}, using first")
            # print(vid)
            all_cal = compute_POG_cal_error(df_cal, phone_model[0])
            # all_sess_cal = pd.concat([all_sess_cal, all_cal]).reset_index(drop=True) # TODO: odd pattern, concat at end
            df_collector.append(all_cal)

        pog_cal_error = pd.concat(df_collector, axis=0).reset_index(drop=True)

    # Merge CV and pog err
    POG_ERR_COLS = [
        #'session_id',
        # 'task_id',
        "video_path",
        "event_order",
        "per_dot_threshold_both2",
        "per_dot_threshold_both3",
        "per_dot_threshold_y3",
        "per_dot_threshold_y2",
        "pog_error_x",
        "pog_error_y",
        "pog_x_std",
        "pog_y_std",
    ]
    df2 = df.merge(
        pog_cal_error[POG_ERR_COLS], on=["video_path", "event_order"], how="left"
    )

    # nans added is just from new POG_ERR_COLS stretched over, no problem
    # will also have nans for tasks that aren't cal
    assert (df2.isna().sum().sum() - df.isna().sum().sum()) == df2[
        POG_ERR_COLS
    ].isna().sum().sum()

    # Check
    print(df2.shape, df2["session_id"].nunique())
    label_dist(df2, col="ptsd_bin")

    # Add complianace stats columns
    if not args.bypass_comp:
        print("Adding compliance statistics...")
        if not os.path.exists(args.compliance_stats):
            print(
                f"Warning: Compliance stats file not found at {args.compliance_stats}"
            )
            print("Skipping compliance stats merge.")
        else:
            try:
                dfs = pd.read_parquet(args.compliance_stats)
                if "video_filename" in dfs.columns:
                    del dfs["video_filename"]
                print(
                    f"Unique videos in compliance stats: {dfs['vid_basename'].nunique()}"
                )
                df2["vid_basename"] = df2["video_path"].apply(
                    lambda x: os.path.basename(x)
                )  # for better matching

                # NOTE: Will merge based on what's queried (latest vids as they come in)
                # Any vids not run by compliance script will have nans in compliance
                # columns if `compliance_stats.parquet` isn't up to date
                df2 = df2.merge(dfs, on="vid_basename", how="left")
            except Exception as e:
                print(
                    f"Error loading compliance stats from {args.compliance_stats}: {e}"
                )
                print("Continuing without compliance stats.")

    ## Final Check
    print("=" * 20)
    print("Final check:")
    print(df2.shape, df2["session_id"].nunique())
    label_dist(df2, col="ptsd_bin")

    if args.save:
        save_df(df2, args, os.getcwd())


if __name__ == "__main__":
    parser = ArgumentParser(description="Get data for yochlol project.")
    parser.add_argument(
        "--filter_cohorts",
        type=int,
        nargs="+",
        default=None,
        help="example: `1` or `1 4` to select cohorts",
    )
    parser.add_argument(
        "--compliance-stats",
        default="compliance_stats.parquet",
        help="Path to compliance statistics parquet file (default: compliance_stats.parquet)",
    )
    parser.add_argument(
        "--bypass-comp",
        action="store_true",
        help="Bypass merging with external compliance stats",
    )
    parser.add_argument(
        "--split-pcl",
        action="store_true",
        help="Remove intermediate PCL scores.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the dataframe to parquet (default: working dir).",
    )
    args = parser.parse_args()

    # Convert to single int if only one value given
    if args.filter_cohorts is not None and len(args.filter_cohorts) == 1:
        args.filter_cohorts = args.filter_cohorts[0]

    main(args)
