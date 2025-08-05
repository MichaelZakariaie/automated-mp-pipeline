# New ML Pipeline Naming Convention

## Overview
The ML pipelines now use a persistent counter system for output folder naming:
- Format: `run_X_YYYYMMDD_HHMMSS`
- X = incrementing counter (persists across sessions)
- Example: `run_4_20250717_125048`

## Key Features

### 1. Persistent Counter
- Counter stored in `.ml_run_counter.json`
- Automatically increments with each run
- Thread-safe implementation prevents conflicts

### 2. Unified Output Structure
All outputs (regular or k-fold) now save to a single folder:
```
ml_output/
└── run_5_20250717_125202/
    ├── report.html              # Single aggregated HTML report
    ├── kfold_metadata.json      # (k-fold) or run_metadata.json (regular)
    ├── kfold_results_summary.csv
    ├── kfold_detailed_results.csv
    ├── kfold_model_comparison.png
    ├── kfold_score_distributions.png
    ├── kfold_confusion_matrices.png
    └── kfold_roc_curves.png
```

### 3. Auto-Detection
- Report generator automatically detects pipeline type
- No need to specify report type manually
- Based on presence of `kfold_metadata.json` vs `run_metadata.json`

### 4. Simplified File Names
- Removed target column from file names
- Each run has its own folder, so no naming conflicts
- Cleaner, more consistent structure

## Usage

### Running Pipelines
```bash
# Regular pipeline
python ml_pipeline_with_pcl.py --target ptsd_bin

# K-fold pipeline  
python ml_pipeline_kfold.py --target ptsd_bin --n-iterations 5

# With PCL filtering
python ml_pipeline_with_pcl.py --target ptsd_bin --remove-intermediate-pcl
```

### Viewing Reports
```bash
# List all reports (sorted by run number)
python view_reports.py -l

# View specific report
python view_reports.py 1
```

### Report Server
- Automatically starts when pipelines complete
- Access at: http://localhost:8888/run_X_timestamp/report.html
- Use VS Code port forwarding for remote access

## Benefits
1. **Persistent Numbering** - Easy to track runs chronologically
2. **Single Folder** - All outputs for a run in one place
3. **Cleaner Names** - No redundant target info in filenames
4. **Auto-Detection** - Smart report generation
5. **Better Organization** - Clear run separation