import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def aggregate_compliance_stats(json_dir):

    # Define exactly which nested stats to grab:
    fields = [
        ("orientation_stats", "roll"),
        ("orientation_stats", "pitch"),
        ("orientation_stats", "yaw"),
        ("position_stats", "z_depth"),
        ("velocity_stats", "roll"),
        ("velocity_stats", "pitch"),
        ("velocity_stats", "yaw"),
        ("velocity_stats", "total"),
        ("face_size_stats", "percentage"),
    ]

    # Extract into a list of flat dicts:
    rows = []
    for fname in tqdm(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        fullpath = os.path.join(json_dir, fname)
        with open(fullpath, "r") as f:
            data = json.load(f)

        base = data["algorithm_results"]["FaceMeshIMUAnalysis"]
        row = {"video_filename": os.path.basename(fname), "face_detected": None}

        # Check if no face was detected
        if "error" in base:
            print(f"No face detected in {fname}: {base['error']}")
            row["face_detected"] = 0
            # Set all stats to NaN
            for category, stat in fields:
                row[f"{category}_{stat}_mean"] = np.nan
                row[f"{category}_{stat}_std"] = np.nan
        else:
            row["face_detected"] = 1
            # Normal processing
            for category, stat in fields:
                try:
                    stats = base[category][stat]
                    # create two columns per field:
                    row[f"{category}_{stat}_mean"] = stats["mean"]
                    row[f"{category}_{stat}_std"] = stats["std"]
                except KeyError as e:
                    print(f"Warning: Missing key {e} in {fname}")
                    print(f"Available keys in base: {list(base.keys())}")
                    if category in base:
                        print(
                            f"Available keys in {category}: {list(base[category].keys())}"
                        )
                    # Set default values
                    row[f"{category}_{stat}_mean"] = np.nan
                    row[f"{category}_{stat}_std"] = np.nan

        # Add ambient light
        # Load & account for json vs csv bug that calls it 'unified' analysis
        csv_name = os.path.splitext(fname)[0] + ".csv"
        csv_name = csv_name.split("_")
        csv_name.insert(-3, "unified")
        csv_name = "_".join(csv_name)
        # get & avg ambient light
        cdf = pd.read_csv(os.path.join(json_dir, csv_name))
        light_mean = cdf["BasicVideoAnalysis_brightness"].mean()
        light_std = cdf["BasicVideoAnalysis_brightness"].std()
        row["basic_brightness_mean"] = light_mean
        row["basic_brightness_std"] = light_std
        rows.append(row)

    # Build df
    df = pd.DataFrame(rows)
    # print(df.head())
    # print(df.shape)
    # print(df.columns.tolist())
    # print(df.isna().sum())
    return df


def fix_json_fname_format(fname: str, add_mp4_ext=False):
    """make the format output from compliance tool match s3 videopath"""
    fname = os.path.splitext(fname)[0]
    # remove "_analysis_{date}_{time}" from end of json to get matching vidname
    fname = fname.split("_")[:-3]
    fname = "_".join(fname)
    if add_mp4_ext:
        fname = fname + ".mp4"
    return fname


def main(args):
    # most compliance stuff, ambient light, etc.
    df_comp = aggregate_compliance_stats(args.json_dir)
    print(f"df_comp shape {df_comp.shape}")

    # merge
    # TODO: unify format of video_filepath
    # '04138ed1-16ae-44af-b321-2177e80ae565_1749674822584_plr_view_analysis_20250702_132727.json'

    # df_comp = df_comp.iloc[0]
    df_comp["vid_basename"] = df_comp["video_filename"].apply(
        lambda x: fix_json_fname_format(x, add_mp4_ext=True)
    )

    print(df_comp.shape)
    print(df_comp["vid_basename"].shape)

    # Save
    if args.save_path is None:
        cwd = os.getcwd()
        fname = "compliance_stats.parquet"
        fname = os.path.join(cwd, fname)
        print(f"Saving stats to {fname}")
        df_comp.to_parquet(fname)
    else:
        fname = "compliance_stats.parquet"
        fname = os.path.join(args.save_path, fname)
        print(f"Saving stats to {fname}")
        df_comp.to_parquet(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        type=str,
        default="/media/m/mp_compliance/tmp/metrics",
        help="path to json/csv output from unified_compliance_analyzer.py",
    )
    parser.add_argument("--save_path", type=str, default=None, help="savepath")
    args = parser.parse_args()
    main(args)
