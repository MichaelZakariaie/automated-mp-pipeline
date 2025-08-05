from typing import Tuple

import awswrangler as wr

# import malfoy.insight_config as insight_config
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# from malfoy.data_loader import insight_prep
from matplotlib import pyplot as plt

# from scipy import signal
from scipy.signal import resample
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tsai.all import *
from tsai.data import validation
from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features
from tslearn.utils import to_time_series_dataset
from wrangle_ts import define_unique_groups


# MiniROCKET functions
def get_tsdfs(unique_items, dfg, tscols):  # categorical=False):
    """Returns np.ndarray of time series given gropuby obj and unique items.

    Args:
    -----
    unique_items : list of uniques, like session, video path, etc.
    dfg : pd.DataFrame groupby object associated with unique items
    tscols : list of df column names to include
    categorical : [DEPRECATED] whether to process categorical or not

    Returns:
    --------
    x : numpy array of time series

    """
    df_assembler = []

    for u in unique_items:
        tmp = dfg.get_group(u).reset_index(drop=True)
        df_assembler.append(tmp[tscols])
        # append time series or categorical columns
        # categorical for rocket feats, need to factorize if so
        # df_assembler.append(tmp[tscol] if not categorical else tmp[ycols])
    # to_time_series_dataset() will put everything together into
    # a square array for you by padding everything to the length
    # of the longest single sequence
    # but check.....might add a ton of 0 padding
    tsd = to_time_series_dataset(df_assembler)
    x = torch.from_numpy(tsd).permute(0, 2, 1).numpy()
    return x


def get_mrfs(X, splits, num_kernels=10000):
    """Get MiniRocketFeatures in one convenient function.

    Args:
    -----
    X : data in, torch.Tensor of shape (batchsize, channels, length)
    splits : tuple, split idx for data, from tsai.data.validation.get_splits()
    num_kernels : int, desired number of rocket kernels

    Returns:
    --------
    X_feat : np.ndarray, minirocket features of shape (batchsize, rocket kernels, 1)
    mrf : fitted MiniRocketFeatures model instance

    """
    # mrf = MiniRocketFeatures(xXx.shape[1], xXx.shape[2]).to(torch.device('cuda:0'))
    mrf = MiniRocketFeatures(X.shape[1], X.shape[2], num_features=num_kernels).to(
        torch.device("cuda:0")
    )
    X_train = X[splits[0]]
    mrf.fit(X_train, chunksize=128)
    X_feat = get_minirocket_features(X, mrf, chunksize=64, to_np=True)
    return X_feat, mrf


def run_minirocket(
    df,
    timeseries_columns=None,
    target_col="ptsd_bin",
    num_rkt_kernels=10000,
    chunksize=512,
) -> Tuple[pd.DataFrame, pd.Index]:
    """

    df : dataframe
    timeseries_columns : list of strings
        timeseries_columns should be timeseries only, not categorical or non-timeseries cols
    num_rkt_kernels : int
    """
    print(f"\nRunning MiniROCKET feature transform...")

    if timeseries_columns is None:
        print(f"No timeseries_columns provided, using default")

    # Get unique vids and group dataframe
    vids, grp = define_unique_groups(df, column="video_filename")

    # Rocket can't take a ragged array
    # should alread be equal length after interpolation, but prep
    X_for_rocket = get_tsdfs(
        vids, grp, timeseries_columns
    )  # nparray -> (1115, 7, ,2394) == (videos, features, time)
    X = X_for_rocket.astype(np.float32)

    # split the data (not really required for ROCKET transforms but best practice)
    splits = validation.get_splits(vids, valid_size=0.1)

    # Instantiate rocket & "fit"
    mrf = MiniRocketFeatures(X.shape[1], X.shape[2], num_features=num_rkt_kernels).to(
        default_device()
    )

    X_train = X[splits[0]]

    try:
        mrf.fit(X_train, chunksize=chunksize)
        # NOTE: mrf.fit() doesn't actually train anything
    except RuntimeError as e:
        print(e)
        print("Try decreasing minirocket chunksize kwarg")
        # Why? chunksize=512 works for (2624, 6, 2316) = (videos, features, timesteps)
        # but if yours is much larger (more feats, tasks, etc.),
        # this might be too much for pytorch to handle in 1 chunk of the given size
        #
        # RuntimeError: Expected canUse32BitIndexMath(input) && canUse32BitIndexMath(output) to be true, but got false.
        # https://github.com/pytorch/pytorch/issues/80020

    # Get rocket features
    X_feat = get_minirocket_features(X, mrf, chunksize=chunksize, to_np=True)

    # X_feat.shape, type(X_feat)
    rocket_column_end = X_feat.shape[0]
    print(f"MiniROCKET features output shape: {X_feat.shape}")

    # Add back labels & other info to Rocketed vids.
    # shape: (videos, rocket_features + other info)
    rocket_df = pd.DataFrame(
        X_feat.squeeze(), columns=[str(i) for i in range(X_feat.shape[1])]
    )  # shape = (videos, rocket_features)

    # Map appropriate columns back on after rocketing
    rocket_df["video_filename"] = (
        vids  # NOTE: should match based on order, this is critical!
    )
    for column in ["unique_id", "session_id", target_col]:
        column_map = dict(zip(df["video_filename"], df[column]))
        rocket_df[column] = rocket_df["video_filename"].map(column_map)

    rocket_df[target_col] = rocket_df[target_col].astype(int)

    # ID rocket cols for later model loading
    rocket_columns = rocket_df.iloc[:, :rocket_column_end].columns
    print(f"Rocketed features df shape: {rocket_df.shape}")
    return rocket_df, rocket_columns
