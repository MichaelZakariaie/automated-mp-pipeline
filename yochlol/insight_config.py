# TODO: To be moved to config file later (ini/.yaml/.json) via configparser

###################################
# Script Behavior
VERBOSE = True

DATASET = None  # dataset source other than from Athena query like "/my/path.parquet"

INSPECT = False  # if True, saves the dataset after prep pipeline for further inspection
INSPECT_SAVEPATH = "~/"

TRAIN = True  # train ML models; use False to stop after prep, debugging, etc.
HOLDOUT = False  # hold out test for final generalization, beyond cross-val

SHOW_RESULTS = True  # make visiualizations for model scores

OUTPATH = "."  # path to save plots/images (. == hogwarts)


###################################
# Data Ingestion Variables

# COHORT = "all"
COHORT = 5
# COHORT = [4, 5]

# QUERY = "SELECT DISTINCT * FROM cv"
# DATABASE = "ptsd_ios"  # UT, 'ptsd_pilot' # UCLA
# SITE_PREFIXES = ["se10", "u1", "u2", "u4", "u5", "u6", "u8", "u9"]  # ["u1", "se1"]
# TEST_SAMPLES = "n_"  # multiple substrings should be 1 expression like "n_|test"
# PTSD_LABELS_FILE = "/data/datasets/prepped/ptsd/ptsd_labels_165.tsv"
# EXCLUSIONS_FILE = "/data/datasets/prepped/ptsd/exclusions_981.tsv"

# Max acceptable dropped frames in CV timeseries beyond which to throw out data
# EXCLUSION_THRESHOLD = 0.5


###################################
# Features & Model Training Variables

# TARGET = "ptsd_bin"  # 'raps_binary_ptsd', "pcl_score", "raps_weighted_pcl", "raps+pcl", "raps_critE_bin"
TARGET = "raps_binary_ptsd"  # "ptsd_bin"  # "raps_critE_bin"
REMOVE_INTERMEDIATE_LABELS = False

TASK = ["face_pairs"]  # TODO: add support for parsing task here
# TASK = ["face_pairs", "plr", "calibration_1", "mckinnon"]
# TASK = ["calibration_1"]
# TASK = ["calibration_1", "calibration_2"]
# TASK = ["face_pairs", "plr"]
# TASK = ["face_pairs", "plr", "calibration_1", "calibration_2", "mckinnon"]
# TASK = "mckinnon"
# TASK = ["plr"]
# TASK = "heart_rate_calibration" / "heart_rate_baseline" # replace

SEG_FEATS = [
    # "pupil_area_left",
    # "pupil_area_right",
    # "iris_area_left",
    # "iris_area_right",
    # 'iris_centers_left_y',
    # 'iris_centers_left_x',
    # 'iris_centers_right_y',
    # 'iris_centers_right_x',
    # "pupil_centers_right_y",
    # "pupil_centers_right_x",
    # "pupil_centers_left_y",
    # "pupil_centers_left_x",
    "pirl",
    "pirr",
    # 'ipd',
    # 'iid'
]
GAZE_FEATS = [
    # 'gaze_pupil_centers_left_y',
    # 'gaze_pupil_centers_left_x',
    # 'gaze_pupil_centers_right_y',
    # 'gaze_pupil_centers_right_x',
    # "gaze_left_pitch",
    # "gaze_left_yaw",
    # "gaze_right_pitch",
    # "gaze_right_yaw",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "pog_x_coord",
    "pog_y_coord",
]

# FEATS = SEG_FEATS
# FEATS = GAZE_FEATS
FEATS = SEG_FEATS + GAZE_FEATS
# FEATS = SEG_FEATS + GAZE_FEATS + CAT_FEATS  # TODO

USE_CAT_FEATS = True
CAT_FEATS = [
    "task_id_renamed"
]  # TODO: add others from encoder, task-dependent. prob put in a function, here use USE_CAT_FEATS = boolean


UTIL_COLS = [
    "unique_id",
    "session_id",
    #'order_uuid',
    #'video_path',
    "video_filename",
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
    "raps_score",
    "raps+pcl",
    "raps_weighted_pcl",
]

# Data imbalance handling
BALANCE_CLASSES = False  # if True, randomly undersample people from the dataset


DEBLINK = False

RUN_ROCKET = True
NUM_ROCKET_FEATS = 10000  # default, could try 20,000, 100, etc....

RANDOM_SEED = 204542  # 6
CV_TYPE = "group"  # "logo"
NUM_FOLDS = 5
CV_GROUP = "factorized_unique_id"  # column for LOGO/GroupKFold


###################################
# Models to Train

# sklearn baseliners
# MODELS = None
MODELS = ("dummy", "rf", "ridge", "logreg", "svc", "lgbm", "catboost")
# MODELS = ("dummy", "ridge", "logreg")
# MODELS = ("dummy", "rf", "ridge", "logreg", "lgbm", "catboost")
# MODELS = ("dummy", "ridge", "logreg", "catboost")
# MODELS = ("dummy", "catboost")
# MODELS = None
# 1NN-DTW
# Gradient boosted trees: LGBM, XGBoost, CatBoost, ...
# tsai methods (InceptionTime et al)
# etc.

AUTOML = False
