import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

cols = [
    "task_id",
    "event_name",
    "event_type",
    "condition",
    "top_emotion",
    "bottom_emotion",
    "cue_location",
    "dot_location",
    "incongruency",
    "engage_disengage",
]

# time wtf: row_number, timestamps can be out of order,.... why?

# TODO:
# / handcrafted
# / one hot
# / assign to df wrangling
# - match prod col names/organization

various_time_cols = [  # for reference. delete once built
    "frame_timestamps",
    "frametimestamps",
    "video_frametimestamps",
    "row_number",
]

ONEHOT_MAP = {  # for ref
    "task": [0, 1, 2, 3],  # cal, plr, mab, mck
    "mab_engagement / mck_phase": [0, 1, 2],
    "mab_emotion / mck_emotion": [0, 1, 2, 3],
    "mab_emo_location / mck subcat": [0, 1],
    "mab_dot_location": [0, 1],
    "mab_cue_location": [0, 1],
}


def encode_task_handcrafted(df: pd.DataFrame, task="mab") -> np.array:
    df = df[df["task"] == task].copy()
    print(df)
    n_rows = df.shape[0]

    # maps
    MAB_MAP = {
        "engagement": {"disengagement": 0, "engagement": 1},
        "emotion": {"neutral": 0, "happy": 1, "sad": 2, "angry": 3},
        "emo_location": {"bottom": 0, "top": 1},
        "dot_location": {"bottom": 0, "top": 1},
        "cue_location": {"bottom": 0, "top": 1},
    }

    MCKINNON_MAP = {
        "phase": {"fixation": 0, "image": 1, "blank": 2},
        "emotion": {"positive": 0, "negative": 1, "neutral": 2, "negative_arousal": 3},
        "subcategory": {},
    }

    # checks
    def check_length(encoding, n_rows):
        return encoding.shape[0] == n_rows

    # parse task
    match task:
        case "calibration":
            enc = np.zeros(6, dtype=np.uint8)
            enc = np.tile(enc, (n_rows, 1))
            assert check_length(enc, n_rows)
            return enc

        case "plr":
            enc = np.zeros(6, dtype=np.uint8)
            enc[0] = 1
            enc = np.tile(enc, (n_rows, 1))
            assert check_length(enc, n_rows)
            return enc

        case "mab":
            enc = np.zeros(6, dtype=np.uint8)
            enc[0] = 2
            enc = np.tile(enc, (n_rows, 1))
            enc[:, 1] = df["mab_engagement"].apply(lambda x: MAB_MAP["engagement"][x])
            enc[:, 2] = df["mab_emotion"].apply(lambda x: MAB_MAP["emotion"][x])
            enc[:, 3] = df["mab_emo_location"].apply(
                lambda x: MAB_MAP["emotion_location"][x]
            )
            enc[:, 4] = df["mab_dot_location"].apply(
                lambda x: MAB_MAP["dot_location"][x]
            )
            enc[:, 5] = df["mab_cue_location"].apply(
                lambda x: MAB_MAP["cue_location"][x]
            )
            assert check_length(enc, n_rows)
            return enc

        case "mckinnon":
            enc = np.zeros(6, dtype=np.uint8)
            enc[0] = 3
            enc = np.tile(enc, (n_rows, 1))
            enc[:, 1] = df["mck_phase"].apply(lambda x: MCKINNON_MAP["phase"][x])
            enc[:, 2] = df["mck_emotion"].apply(lambda x: MCKINNON_MAP["emotion"][x])
            # enc[:, 3] = df["mck_subcategory"].apply(lambda x: MCKINNON_MAP["subcategory"][x])
            assert check_length(enc, n_rows)
            return enc

        case _:
            raise ValueError(f"'{task}' task not recognized")


def encode_task_onehot(df: pd.DataFrame, task="mab") -> pd.DataFrame:
    df = df[df["task"] == task].copy()
    n_rows = df.shape[0]

    # checks
    def check_length(encoding, n_rows):
        return encoding.shape[0] == n_rows

    # maps
    MAB_MAP = {
        "engagement": {"disengagement": 0, "engagement": 1},
        "emotion": {"neutral": 0, "happy": 1, "sad": 2, "angry": 3},
        "emotion_location": {"bottom": 0, "top": 1},
        "dot_location": {"bottom": 0, "top": 1},
        "cue_location": {"bottom": 0, "top": 1},
    }

    task_1hot = OneHotEncoder(
        categories=[["calibration", "plr", "mab", "mckinnon"]], sparse_output=False
    )

    # parse task
    match task:
        case "calibration":
            # init 1hot dataframe
            output_columns = pd.DataFrame(columns=["task_1hot"])
            # set task column first
            output = task_1hot.fit_transform(df["task"].values.reshape(-1, 1))
            output = pd.Series(list(output))
            output_columns["task_1hot"] = output
            # reindex to match for merging
            output_columns = output_columns.set_index(df.index)
            assert check_length(output_columns, n_rows)
            return output_columns

        case "plr":
            # init 1hot dataframe
            output_columns = pd.DataFrame(columns=["task_1hot"])
            # set task column first
            output = task_1hot.fit_transform(df["task"].values.reshape(-1, 1))
            output = pd.Series(list(output))
            output_columns["task_1hot"] = output
            # reindex to match for merging
            output_columns = output_columns.set_index(df.index)
            assert check_length(output_columns, n_rows)
            return output_columns

        case "mab":
            encoders = {
                "mab_engagement": OneHotEncoder(
                    categories=[["disengagement", "engagement"]], sparse_output=False
                ),
                "mab_emotion": OneHotEncoder(
                    categories=[["neutral", "happy", "sad", "angry"]],
                    sparse_output=False,
                ),
                "mab_emo_location": OneHotEncoder(
                    categories=[["bottom", "top"]], sparse_output=False
                ),
                "mab_dot_location": OneHotEncoder(
                    categories=[["bottom", "top"]], sparse_output=False
                ),
                "mab_cue_location": OneHotEncoder(
                    categories=[["bottom", "top"]], sparse_output=False
                ),
            }

            # init 1hot dataframe
            col_list = list(encoders.keys())
            col_list.insert(0, "task_1hot")
            output_columns = pd.DataFrame(columns=col_list)

            # set task column first
            output = task_1hot.fit_transform(df["task"].values.reshape(-1, 1))
            output = pd.Series(list(output))
            output_columns["task_1hot"] = output

            # loop through the other hierarchical 1hot encoders and add to columns
            for col, encoder in encoders.items():
                output = encoder.fit_transform(df[col].values.reshape(-1, 1))
                output = pd.Series(list(output))
                output_columns[col + "_1hot"] = output
                del output_columns[col]
            output_columns = output_columns.set_index(df.index)
            assert check_length(output_columns, n_rows)
            return output_columns

        case "mckinnon":
            encoders = {
                "mck_phase": OneHotEncoder(
                    categories=[["fixation", "image", "blank"]], sparse_output=False
                ),
                "mck_emotion": OneHotEncoder(
                    categories=[
                        ["positive", "negative", "neutral", "negative_arousal"]
                    ],
                    sparse_output=False,
                ),
                # "mck_subcategory": OneHotEncoder(
                #     categories=[[""]], sparse_output=False
                # ),
            }

            # init 1hot dataframe
            col_list = list(encoders.keys())
            col_list.insert(0, "task_1hot")
            output_columns = pd.DataFrame(columns=col_list)

            # set task column first
            output = task_1hot.fit_transform(df["task"].values.reshape(-1, 1))
            output = pd.Series(list(output))
            output_columns["task_1hot"] = output

            # loop through the other hierarchical 1hot encoders and add to columns
            for col, encoder in encoders.items():
                output = encoder.fit_transform(df[col].values.reshape(-1, 1))
                output = pd.Series(list(output))
                output_columns[col + "_1hot"] = output
                del output_columns[col]
            output_columns = output_columns.set_index(df.index)
            assert check_length(output_columns, n_rows)
            return output_columns

        case _:
            raise ValueError(f"'{task}' task not recognized")


# Example
if __name__ == "__main__":

    # Build test df
    cal = np.tile("calibration", 3)
    plr = np.tile("plr", 3)
    mab = np.tile("mab", 10)
    list_lrn = np.tile("LL", 4)
    mck = np.tile("mckinnon", 10)

    tasks = np.concat((cal, plr, mab, list_lrn, mck), axis=0)

    # Main cols
    df = pd.DataFrame(
        {
            "uid": np.tile("u1010", 30),
            "session": np.tile("13lk45h", 30),
            "metric": np.random.randn(30),
            "task": tasks,
        }
    )

    # add task event cols
    df["mab_engagement"] = np.concat(
        (
            np.full(6, np.nan),
            np.tile("engagement", 5),
            np.tile("disengagement", 5),
            np.full(14, np.nan),
        )
    )
    df["mab_emotion"] = np.concat(
        (
            np.full(6, np.nan),
            np.tile(["neutral", "happy", "sad", "angry", "neutral"], 2),
            np.full(14, np.nan),
        )
    )
    df["mab_emo_location"] = np.concat(
        (
            np.full(6, np.nan),
            np.tile(["bottom", "bottom", "top", "top", "bottom"], 2),
            np.full(14, np.nan),
        )
    )
    df["mab_dot_location"] = np.concat(
        (
            np.full(6, np.nan),
            np.tile(["top", "bottom", "top", "bottom", "top"], 2),
            np.full(14, np.nan),
        )
    )
    df["mab_cue_location"] = np.concat(
        (
            np.full(6, np.nan),
            np.array(
                [
                    "bottom",
                    "top",
                    "bottom",
                    "top",
                    "bottom",
                    "top",
                    "bottom",
                    "top",
                    "bottom",
                    "top",
                ]
            ),
            np.full(14, np.nan),
        )
    )
    df["mck_phase"] = np.concat(
        (
            np.full(20, np.nan),
            np.tile(["fixation"], 4),
            np.tile("image", 3),
            np.tile("blank", 3),
        )
    )
    df["mck_emotion"] = np.concat(
        (
            np.full(20, np.nan),
            np.tile("positive", 2),
            np.tile("negative", 2),
            np.tile("neutral", 2),
            np.tile("negative_arousal", 2),
            np.tile("positive", 2),
        )
    )

    ##############################################
    # EXAMPLE USE - SETUP

    task_cols = [
        "task",
        "mab_engagement",
        "mab_emotion",
        "mab_emo_location",
        "mab_dot_location",
        "mab_cue_location",
        "mck_phase",
        "mck_emotion",
    ]

    task_df = df[task_cols].copy()
    # print(task_df)

    # df["task_encoding"] = encode_task_handcrafted(task_df, task="task")

    # # ----- HANDCRAFTED -----
    # task_df["task_encoding"] = np.nan
    # task_df["task_encoding"] = task_df["task_encoding"].astype(object)

    # enc_vector = encode_task_handcrafted(task_df, task=TASK)
    # mask = task_df["task"] == TASK
    # assert mask.sum() == enc_vector.shape[0]

    # # Assign row by row (inefficient but ok loop)
    # # for idx, encoding in zip(task_df[mask].index, enc_vector):
    # #     task_df.at[idx, "task_encoding"] = encoding

    # # Pandas idiomatic (better assuming indexes align...)
    # enc_series = pd.Series(list(enc_vector), index=task_df[mask].index)
    # task_df.loc[enc_series.index, "task_encoding"] = enc_series
    # print(task_df)

    # ----- ONEHOT -----
    print(task_df)

    enc_df = encode_task_onehot(task_df, task="mab")
    print(enc_df)

    # NOTE: replace task_df with original df once tested

    # NOTE: MAB & McKinnon only
    for col in enc_df.columns:
        if col not in task_df.columns:
            task_df[col] = pd.NA
    task_df.update(enc_df, overwrite=False, errors="raise")
    print(task_df)
    print(task_df["task_1hot"])

    enc_df = encode_task_onehot(task_df, task="calibration")
    print(enc_df)
    task_df.update(enc_df, overwrite=False, errors="raise")
    print(task_df)
    print(task_df["task_1hot"])

    enc_df = encode_task_onehot(task_df, task="plr")
    print(enc_df)
    task_df.update(enc_df, overwrite=False, errors="raise")
    print(task_df)
    print(task_df["task_1hot"])

    enc_df = encode_task_onehot(task_df, task="mckinnon")
    print(enc_df)
    # NOTE: MAB & McKinnon only
    for col in enc_df.columns:
        if col not in task_df.columns:
            task_df[col] = pd.NA
    task_df.update(enc_df, overwrite=False, errors="raise")
    print(task_df)
    print(task_df[["mck_phase", "mck_emotion", "mck_phase_1hot", "mck_emotion_1hot"]])
    print(task_df["task_1hot"])
