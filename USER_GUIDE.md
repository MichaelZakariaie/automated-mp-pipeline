# Automated MP Pipeline - User Guide

This guide explains how to use the new automated pipeline that combines yochlol (time series analysis) and tabular modeling into a single coordinated system. If you're familiar with the original repos, this will help you understand the new integration.

## What's Changed

### Before (Original Repos)
- **yochlol**: Run manually, pull data from AWS, do time series analysis
- **tabular_modeling**: Run separately, pull different data, do ML analysis
- **Manual coordination**: You had to run them separately and coordinate between them

### After (This Pipeline)
- **Unified system**: One pipeline runs both automatically
- **Cohort-aware**: Easily switch between different cohorts/data sources
- **Two-command operation**: Just two commands to run everything
- **Multi-environment**: Each component runs in its own environment (no more dependency conflicts!)

## Quick Start

### 1. Initial Setup (One Time Only)

```bash
# Clone/navigate to the pipeline directory
cd automated-mp-pipeline

# Set up all virtual environments and dependencies
./setup_multi_env.sh
```

This creates 4 virtual environments:
- `env_cv/` - Computer vision processing
- `env_yochlol/` - Time series analysis (your familiar yochlol code)
- `env_tabular/` - Tabular modeling (your familiar tabular code)  
- `env_main/` - Pipeline orchestration

### 2. Two-Command Operation

**Command 1: Run Time Series Analysis**
```bash
./run_time_series_multi.sh
```

This will:
- Pull time series data from AWS (using yochlol)
- Run compliance analysis
- Generate time series models and predictions
- Save results for the external team

**Command 2: Run Tabular Analysis** (after external team processes data)
```bash
./run_tabular_multi.sh
```

This will:
- Pull tabular data from AWS 
- Run all the ML models (Random Forest, XGBoost, etc.)
- Generate comprehensive reports

### 3. Check Results

Results are saved in:
- `pipeline_output/time_series_results/` - Time series analysis results
- `pipeline_output/tabular_results/` - ML model results and reports
- `pipeline_reports/` - Combined final reports

## Working with Different Cohorts

### Understanding Cohorts

Each cohort has different:
- S3 bucket paths
- Database table names  
- Available tasks (face_pairs, calibration, etc.)
- File naming patterns
- Trial counts per task

### Switching Cohorts

**Option 1: Command line (temporary)**
```bash
./run_time_series_multi.sh --cohort cohort_6
./run_tabular_multi.sh --cohort cohort_6
```

**Option 2: Edit config file (permanent)**
Edit `pipeline_config.yaml`:
```yaml
pipeline:
  cohort: cohort_6  # Change this line
```

**Option 3: Add new cohort**
Edit `cohort_config.yaml` and add your cohort configuration (see examples in the file).

### Current Available Cohorts
- `default` - Original hardcoded values (cohort 5)
- `cohort_6` - Example new cohort configuration

## Advanced Usage

### Running Individual Stages

Instead of the two-command operation, you can run individual stages:

```bash
# Just computer vision (when we implement local CV)
source env_main/bin/activate
python multi_env_pipeline.py --stage cv

# Just time series analysis
source env_main/bin/activate  
python multi_env_pipeline.py --stage time_series

# Just tabular analysis
source env_main/bin/activate
python multi_env_pipeline.py --stage tabular
```

### Using the Original Scripts Directly

The original yochlol and tabular_modeling code still works exactly as before. You can:

```bash
# Use yochlol directly
source env_yochlol/bin/activate
cd yochlol/
python get_data.py --save
python wrangle_ts.py --data-path your_data.parquet --save
# etc.

# Use tabular modeling directly  
source env_tabular/bin/activate
cd tabular_modeling/
python download_s3_sessions.py
python ml_pipeline_with_pcl.py --target ptsd_bin
# etc.
```

### Checking Pipeline Status

```bash
source env_main/bin/activate
python run_pipeline.py status
```

This shows:
- Which stages have been completed
- What output files exist
- Current configuration

## Configuration

### Main Configuration (`pipeline_config.yaml`)

Key settings you might want to change:

```yaml
pipeline:
  mode: aws  # or 'local_cv' for local computer vision
  cohort: default  # which cohort to use

computer_vision:
  use_dummy_data: true  # for testing without real videos
  dummy_sessions_count: 10  # how many test sessions

yochlol:
  compliance_metrics: true  # whether to download compliance videos
  task: face_pairs  # which task to focus on

tabular_modeling:
  targets: [ptsd_bin, pcl_score]  # which ML targets to predict
  sample_fraction: 1.0  # use all data (0.1 = 10% for testing)
  use_rfecv: true  # feature selection
  hyperparameter_tuning: true  # optimize model parameters
```

### Cohort Configuration (`cohort_config.yaml`)

This is where cohort-specific settings live. For a new cohort, you need:

```yaml
cohorts:
  your_new_cohort:
    cohort_id: 7
    description: "Your cohort description"
    
    s3:
      compliance_bucket: "senseye-ptsd"
      compliance_prefix: "public/ptsd_ios/cohort_7/"
      data_bucket: "senseye-data-quality"
      data_prefix: "cohort_7_saturn_uploads/"
    
    athena:
      database: "data_quality"
      tables:
        pcl_scores: "mp_pcl_scores_cohort7"  # your PCL table
    
    tasks:
      available: ["face_pairs", "calibration_1", "new_task"]
      default: ["face_pairs"]
      sample_counts:
        face_pairs: 140
        new_task: 60
```

## Testing and Verification

### Test the Setup

```bash
# Test that all environments work
source env_main/bin/activate
python test_multi_env_pipeline.py
```

### Test with Dummy Data

```bash
# Generate some test data
source env_cv/bin/activate
python cv_data_generator.py --sessions 3 --output-dir test_output

# This creates realistic dummy data in the same format the pipeline expects
```

### Check Individual Environments

```bash
# Test yochlol environment
source env_yochlol/bin/activate
python -c "import torch, matplotlib, flaml; print('Yochlol env OK')"

# Test tabular environment  
source env_tabular/bin/activate
python -c "import sklearn, xgboost, pandas; print('Tabular env OK')"
```

## Troubleshooting

### Common Issues

**"Virtual environments not found"**
- Run `./setup_multi_env.sh` first

**"AWS credentials not configured"**
- Run `aws configure` 
- Or check `cd tabular_modeling && python test_aws_connection.py`

**"No cohort configuration found"**
- Check that your cohort name exists in `cohort_config.yaml`
- Make sure you're using the exact name (case-sensitive)

**"Time series files not found"**
- Make sure you ran the time series stage first
- Check that AWS data is available for your cohort

### Checking Logs

All execution is logged to `pipeline_execution.log`. Check this file for detailed error messages.

### Getting Help

1. Check the logs: `tail -f pipeline_execution.log`
2. Test individual environments as shown above
3. Try with a smaller sample: set `sample_fraction: 0.1` in config
4. Run individual stages to isolate the problem

## What's Under the Hood

### How It Works

1. **Multi-Environment Execution**: Each component (yochlol, tabular) runs in its own virtual environment with the right dependencies
2. **Environment Variable Passing**: The orchestrator sets up the right environment variables for each cohort
3. **File Coordination**: Results from one stage are automatically available to the next
4. **Process Management**: Uses subprocess calls to run scripts in the right environments

### Directory Structure

```
automated-mp-pipeline/
├── yochlol/                    # Original yochlol code (unchanged)
├── tabular_modeling/           # Original tabular code (unchanged)
├── env_cv/                     # CV virtual environment
├── env_yochlol/               # Yochlol virtual environment  
├── env_tabular/               # Tabular virtual environment
├── env_main/                  # Main orchestration environment
├── pipeline_config.yaml       # Main settings
├── cohort_config.yaml         # Cohort-specific settings
├── multi_env_pipeline.py      # Main orchestration script
├── run_time_series_multi.sh   # Simple command 1
├── run_tabular_multi.sh       # Simple command 2
└── pipeline_output/           # All results go here
```

### Key Innovation

The main innovation is **environment isolation**. Instead of trying to make all dependencies work together (which caused conflicts), each component runs in its own environment. The orchestrator coordinates between them using:

- Subprocess calls with the right Python interpreter
- Environment variable passing for cohort settings
- File system coordination for data passing
- Logging and error handling across environments

This gives you all the power of the original repos while making them work together seamlessly.

## Migration from Original Workflow

### If You Were Using yochlol Before

Instead of:
```bash
cd yochlol/
python get_data.py --save
python wrangle_ts.py --data-path data.parquet --save
python train.py --data-path wrangled.parquet
```

Now just:
```bash
./run_time_series_multi.sh
```

### If You Were Using tabular_modeling Before  

Instead of:
```bash
cd tabular_modeling/
./run_full_pipeline.sh
```

Now just:
```bash
./run_tabular_multi.sh
```

### If You Were Using Both

Instead of running them separately and coordinating manually, now just:
```bash
./run_time_series_multi.sh
# Wait for external team...
./run_tabular_multi.sh
```

The pipeline handles all the coordination, environment management, and cohort configuration for you.