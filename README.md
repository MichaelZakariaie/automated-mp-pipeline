# Automated MP Pipeline

A unified pipeline that orchestrates computer vision processing, time series analysis (yochlol), and tabular modeling for PTSD prediction from video data.

## Overview

This pipeline automates the entire workflow from raw videos to final analysis reports:

1. **Stage 1**: Computer Vision / Time Series Data Generation
   - Mode 1 (AWS): Pull existing time series data from AWS
   - Mode 2 (Local CV): Process videos locally with CV algorithms (currently uses dummy data)

2. **Stage 2**: Time Series Analysis (yochlol)
   - Process time series data
   - Generate compliance metrics
   - Run ROCKET models for time series classification

3. **Stage 3**: Tabular Analysis
   - Process session data into tabular format
   - Merge with PCL scores
   - Run ML models for PTSD prediction

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Configure AWS (if using AWS mode)
aws configure
```

### Two-Command Operation

The pipeline is designed to run with just two commands:

#### Command 1: Run Time Series Analysis
```bash
python run_pipeline.py time_series
```

This will:
- Generate or fetch time series data (based on mode)
- Run time series analysis using yochlol
- Save results for the external team

#### Command 2: Run Tabular Analysis (after external team is done)
```bash
python run_pipeline.py tabular
```

This will:
- Download tabular data from AWS
- Run tabular modeling analysis
- Generate final reports

### Check Status
```bash
python run_pipeline.py status
```

## Configuration

### Main Configuration

Edit `pipeline_config.yaml` to control the pipeline behavior:

### Switch Between AWS and Local CV Mode
```yaml
pipeline:
  mode: aws  # Change to 'local_cv' for local processing
  cohort: default  # Specify which cohort configuration to use
```

### Cohort Configuration

The pipeline supports multiple cohort configurations defined in `cohort_config.yaml`. This allows you to:
- Use different S3 buckets and paths for different cohorts
- Configure different tasks and trial counts
- Specify cohort-specific file naming patterns
- Use different database tables

#### Using a Specific Cohort
```bash
# Option 1: Specify cohort in command line
python run_pipeline.py time_series --cohort cohort_6

# Option 2: Update pipeline_config.yaml
# Set cohort: cohort_6 in the file

# Option 3: Update hardcoded values in scripts
python update_cohort_scripts.py --cohort cohort_6
```

#### Adding a New Cohort

Edit `cohort_config.yaml` and add your cohort configuration:

```yaml
cohorts:
  my_new_cohort:
    cohort_id: 7
    description: "Cohort 7 - Fall 2024"
    s3:
      compliance_bucket: "senseye-ptsd"
      compliance_prefix: "public/ptsd_ios/cohort_7/"
      data_bucket: "senseye-data-quality"
      data_prefix: "cohort_7_uploads/"
    athena:
      database: "data_quality"
      tables:
        pcl_scores: "mp_pcl_scores_cohort7"
    tasks:
      available: ["face_pairs", "new_task"]
      default: ["face_pairs"]
```

### Configure Computer Vision (for future local processing)
```yaml
computer_vision:
  use_dummy_data: true  # Set to false when real CV is implemented
  dummy_sessions_count: 10
  video_input_dir: ./input_videos  # Where to find videos
```

### Configure ML Models
```yaml
tabular_modeling:
  models:
    - Random Forest
    - XGBoost
    - Neural Network
  use_rfecv: true
  hyperparameter_tuning: true
```

## Directory Structure

```
automated-mp-pipeline/
├── yochlol/                    # Time series analysis submodule
├── tabular_modeling/           # Tabular ML submodule
├── pipeline_config.yaml        # Main configuration
├── main_pipeline.py           # Core orchestration logic
├── run_pipeline.py            # Simple CLI interface
├── cv_data_generator.py       # Dummy CV data generator
└── pipeline_output/           # Generated output
    ├── time_series_results/   # Time series analysis results
    ├── tabular_results/       # Tabular modeling results
    └── reports/               # Final reports
```

## Pipeline Modes

### AWS Mode (Current Default)
- Pulls existing time series data from AWS
- Uses the same workflow as the original yochlol repo
- Suitable for current operations

### Local CV Mode (Future)
- Will process videos locally using computer vision
- Currently generates dummy data for testing
- Prepares for future full automation

## Output

The pipeline generates:

1. **Time Series Results**
   - Processed time series data
   - Compliance metrics
   - ROCKET model predictions

2. **Tabular Results**
   - ML model performance metrics
   - Feature importance analysis
   - ROC curves and confusion matrices

3. **Final Reports**
   - Comprehensive HTML reports
   - Statistical analysis summaries
   - Visualization plots

## Advanced Usage

### Run Full Pipeline (Not Recommended)
```bash
python main_pipeline.py --stage full
```

### Custom Configuration
```bash
python run_pipeline.py time_series --config custom_config.yaml
```

### Generate Dummy CV Data Only
```bash
python cv_data_generator.py --sessions 20 --output-dir test_data
```

## Future Enhancements

1. **Real Computer Vision Implementation**
   - Replace dummy data generator with actual CV algorithms
   - Process raw videos to generate time series data
   - Upload results to AWS for external team

2. **Automated Tabular Data Detection**
   - Poll AWS for new tabular data availability
   - Automatically trigger stage 2 when ready

3. **Enhanced Reporting**
   - Unified report combining all analyses
   - Interactive dashboards
   - Automated insights generation

## Troubleshooting

### AWS Access Issues
```bash
# Test AWS connection
cd tabular_modeling
python test_aws_connection.py
```

### Memory Issues
- Reduce `sample_fraction` in config
- Process fewer sessions at once
- Increase system memory allocation

### Missing Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r yochlol/requirements.txt
pip install -r tabular_modeling/requirements.txt
```

## Notes for Users Familiar with Original Repos

- The pipeline preserves the original structure of yochlol and tabular_modeling
- You can still run individual scripts from each subdirectory if needed
- Configuration files from the original repos are still respected
- The main addition is the orchestration layer that connects everything

## Contact

For issues or questions:
1. Check the execution logs: `pipeline_execution.log`
2. Review individual repo documentation in subdirectories
3. Ensure AWS credentials are properly configured