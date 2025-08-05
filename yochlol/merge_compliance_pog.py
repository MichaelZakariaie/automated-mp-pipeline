import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def main(args):
    # Load stats and pog dfs from args
    dfs = pd.read_parquet(args.compliance_stats)
    dfp = pd.read_parquet(args.pog_data)

    # filter dfp to only be calibration tasks
    dfp = dfp[dfp["task_id_renamed"].isin(["calibration_1", "calibration_2"])]
    print(
        "POG error data is only available for calibration tasks (calibration_1, calibration_2)"
    )
    print(
        "POG error metrics will not be available for other tasks in the compliance data"
    )

    # adjust format of video_path to vid_basename
    dfp["vid_basename"] = dfp["video_path"].apply(lambda x: os.path.basename(x))

    # average dfp pog_error x and y and std by video
    POG_COLS = [
        "vid_basename",
        "pog_error_x",
        "pog_error_y",
        "pog_x_std",
        "pog_y_std",
    ]
    dfp = dfp[POG_COLS]
    dfp_avg = dfp.groupby("vid_basename").mean().reset_index()

    print(f"Unique videos in compliance stats: {dfs['vid_basename'].nunique()}")
    print(
        f"Unique videos in POG data (calibration only): {dfp_avg['vid_basename'].nunique()}"
    )

    # filter dfp vid_basename that are in dfs - inner join
    df3 = pd.merge(dfs, dfp_avg[POG_COLS], on="vid_basename", how="left")

    # rename cols below to prepent "avg_"
    AVG_POG_COLS = [
        "pog_error_x",
        "pog_error_y",
        "pog_x_std",
        "pog_y_std",
    ]
    rename_dict = {col: f"avg_{col}" for col in AVG_POG_COLS}
    df3 = df3.rename(columns=rename_dict)

    # Save the merged dataframe if requested
    if args.save:
        df3.to_parquet("merged_compliance_pog.parquet")
        print(
            f"Saved merged data with {len(df3)} rows (vids) to merged_compliance_pog.parquet"
        )
    else:
        print(f"Processed {len(df3)} rows (vids). Use --save to save output.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Merge compliance statistics with POG error data from calibration sessions"
    )
    parser.add_argument(
        "--compliance-stats",
        default="compliance_stats.parquet",
        help="Path to compliance statistics parquet file (default: compliance_stats.parquet)",
    )
    parser.add_argument(
        "--pog-data",
        required=True,
        help="Path to POG error data in MP data parquet file. File name format: MP_cohort_{cohort}_{num_session}sessions_{timestamp}.parquet",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the merged data to merged_compliance_pog.parquet",
    )
    args = parser.parse_args()

    main(args)
