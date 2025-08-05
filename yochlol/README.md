# pstd/time series data wrangler for messy prototyping
_a quick & dirty version that gets you a working time series dataframe for further processing, depending on your goals_

This is not a completely automated pipline. Extra orchestration required if you want to run step 3 before 2 finishes, or 4+ before step 3 finishes.

# Data Ingestion & Compliance
## Quickstart
```bash
# 0. Decide where to save videos and compliance metric temp files (defaults work but check pre-existing files)

# 1. configure aws
bash aws_throughput_config.sh

# 2. download vids -- ~1+ hour
python3 download_compliance_vids.py # saves vids to .../tmp/videos

# 3. process -- 5+ hours
python3 run_compliance.py --limit <NUM_VIDS> # saves 3 compliance report files per video to .../tmp/metrics dir

# 4. aggregate from jsons/csvs
python3 get_stats.py # -> compliance_stats.parquet

# 5. query & save MP PTSD dataset and combine compliance statistics (saves to current working dir)
python3 get_data.py --save

# 6. delete files when done
```

#### default paths
```python
# vids
DOWNLOAD_PATH = "/media/m/mp_compliance/tmp/videos"
# compliance metrics
METRICS_PATH = "/media/m/mp_compliance/tmp/metrics"
```

# TS wrangling / modeling
```bash
# time series specific wrangling
python3 wrangle_ts.py --data-path </your/data/from/get_data.parquet> --task <your_task> --save
python3 wrangle_ts.py --help

# run rocket
python3 train.py --data-path wrangled_ts_face_pairs_453sessions_30fps_1752744717.parquet --rkt-chunksize 256
```


# other contents
```
ptsd-tidy.ipynb # useful notebook of wrangling functions, cells, and baseline and some SOTA models
encode_task.py  # task embedding (hand-crafted and 1hot)
```
