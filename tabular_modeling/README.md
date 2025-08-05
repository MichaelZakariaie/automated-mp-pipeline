# Tabular ML Modeling Pipeline

A comprehensive machine learning pipeline for tabular data analysis with advanced features including automated hyperparameter tuning, feature selection, and multi-model evaluation. Originally designed for PTSD prediction from session data, but adaptable to any tabular dataset.

## 🚀 New Features (2024)

### Core Capabilities
- **11 ML Models**: From simple baselines (Logistic/Linear Regression) to advanced neural networks
- **Automated ML Pipeline**: One-command RFECV + hyperparameter tuning
- **YAML Configuration**: Control everything via config files, no code changes needed
- **Multi-Target Support**: Run multiple experiments with different targets
- **Advanced Neural Networks**: 4 different architectures (standard, deep, wide, funnel)
- **Non-linear Optimization**: Polynomial SVM and diverse activation functions
- **K-fold Cross-Validation**: With configurable iterations and parallel processing

### Available Models
1. **Logistic Regression** - Linear baseline for classification
2. **Linear Regression** - Linear baseline for regression  
3. **Naive Bayes** - Fast probabilistic model
4. **Random Forest** - Ensemble tree method
5. **Gradient Boosting** - Sequential boosting
6. **XGBoost** - Optimized gradient boosting
7. **SVM** - Support Vector Machine (RBF kernel)
8. **SVM_Poly** - Polynomial kernel SVM for non-linear patterns
9. **Neural Network** - Standard 2-layer network (100→50)
10. **Deep NN** - 4-layer deep network (200→100→50→25) with early stopping
11. **Wide NN** - Single wide layer (300) with tanh activation
12. **Funnel NN** - 5-layer funnel architecture (256→128→64→32→16)

## Quick Start with YAML Configuration

```bash
# Run with the main configuration file
python run_all_variations.py --config configs/six_variations.yml
```

### Key Configuration Options

Edit `configs/six_variations.yml` to control:

```yaml
# Data settings
sample_fraction: 0.05    # Use 5% for testing, 1.0 for full data

# Feature engineering
feature_engineering:
  use_rfecv: true       # Enable automatic feature selection
  rfecv_min_features: 5 

# Model selection  
models:
  include_models:       # Choose specific models
    - 'XGBoost'
    - 'Deep NN'
    - 'Random Forest'

# Hyperparameter tuning
hyperparameter_tuning:
  enabled: true         # Enable automatic tuning
  n_iter: 50           # RandomizedSearchCV iterations
  top_models: 3        # Tune top 3 performers

# Automated pipeline
automated_pipeline:
  enabled: true        # Run RFECV + tuning automatically
```

## Overview (Original)

The pipeline processes face_pairs session data and combines it with PCL (PTSD Checklist) scores to predict PTSD status. Each session is transformed into a single row with 982 features (7 features × 140 trials + 2 session-level features), then merged with PCL scores for ML analysis.

## Requirements

- Python 3.7+
- AWS account with access to:
  - S3 bucket: `senseye-data-quality`
  - Athena database: `data_quality`
  - Table: `mp_pcl_scores`

## Installation

1. Clone/navigate to the repository:
```bash
cd /home/michael/tabular_modeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## AWS Setup

### Configure AWS Credentials

```bash
aws configure
```

Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Default output format (json)

### Test AWS Connection

Before running the pipeline, verify your AWS access:

```bash
python test_aws_connection.py
```

This will check:
- AWS credentials
- S3 bucket access
- Athena database access
- PCL table availability

## Running the Pipeline

### Option 1: Run Full Pipeline (Recommended)

```bash
./run_full_pipeline.sh
```

This will automatically:
1. Download all session files from S3
2. Process them into a single dataframe (982 features per session)
3. Fetch PCL scores from AWS Athena
4. Merge PCL scores with session data
5. Train ML models for both PTSD classification and PCL regression

### Option 2: Run Steps Individually

```bash
# Step 1: Download files from S3
python download_s3_sessions.py

# Step 2: Process the files
python process_all_sessions.py

# Step 3: Fetch and merge PCL scores
python fetch_pcl_scores.py

# Step 4: Run ML models for PTSD prediction
python ml_pipeline_with_pcl.py --target ptsd_bin

# Step 5: Run ML models for PCL score regression
python ml_pipeline_with_pcl.py --target pcl_score
```

## Pipeline Scripts

### Core Scripts

1. **download_s3_sessions.py**
   - Downloads session files from S3 path: `s3://senseye-data-quality/messy_prototyping_saturn_uploads/{sess}/face_pairs/processed_v2/`
   - Uses parallel downloads for efficiency
   - Skips already downloaded files

2. **process_all_sessions.py**
   - Transforms each session into a single row with 982 features
   - Features include: dot_latency_pog, cue_latency_pog, trial_saccade_data_quality_pog, etc.
   - Handles missing trials with NaN values

3. **fetch_pcl_scores.py**
   - Queries AWS Athena for PCL scores
   - Creates binary PTSD labels (Positive=1, Negative=0)
   - Merges scores with processed session data
   - Optional: Remove intermediate PCL scores near threshold

4. **ml_pipeline_with_pcl.py**
   - Trains 5 different ML models:
     - Random Forest
     - Gradient Boosting
     - XGBoost
     - SVM
     - Neural Network
   - Supports both classification (PTSD) and regression (PCL score)
   - Includes feature importance analysis
   - Generates ROC curves and performance metrics

### Utility Scripts

- **test_aws_connection.py** - Verifies AWS access before running pipeline
- **run_full_pipeline.sh** - Bash script to run entire pipeline
- **transform_parquet.py** - Original single-file transformation script

## Output Files

### Data Files
- `downloaded_sessions/` - Raw parquet files from S3 (496 sessions)
- `processed_sessions/all_sessions_processed.parquet` - Processed features without PCL
- `processed_sessions/all_sessions_with_pcl.parquet` - **Final dataset with PCL scores** (481 sessions with PCL, 15 without)
- `processed_sessions/all_sessions_with_pcl.csv` - Same as above in CSV format

### ML Results
The ML pipeline has been successfully run with the following results:

#### PTSD Binary Classification Results:
- **Best Model**: Random Forest (AUC = 0.580)
- Test Accuracies:
  - Random Forest: 57.7%
  - Gradient Boosting: 54.6%
  - XGBoost: 51.6%
  - SVM: 47.4%
  - Neural Network: 53.6%

#### Statistical Analysis Summary:
- **Baseline Performance**: Majority class (always predict positive) = 53.6%
- **Statistical Significance**: NO models performed significantly better than baseline (all p > 0.05)
- **Clinical Utility**: All AUC values < 0.6, indicating poor discriminative ability
- **Conclusion**: The features do not appear to be predictive of PTSD status

#### PCL Score Regression Results:
- **Best Model**: Random Forest (R² = 0.007)
- All models showed overfitting with poor generalization to test set

#### Files Generated:
- `ml_results/` - Trained models and performance metrics
  - `*_model_ptsd_bin.pkl` - Models for PTSD classification
  - `*_model_pcl_score.pkl` - Models for PCL regression
  - `model_results_summary_*.csv` - Performance metrics
- `feature_importance_ptsd_bin.png` - Top 20 important features for PTSD
- `feature_importance_pcl_score.png` - Top 20 important features for PCL
- `ml_results/model_comparison_*.png` - Model performance comparison
- `ml_results/roc_curves.png` - ROC curves for PTSD classification
- `ml_results/confusion_matrices.png` - Confusion matrices
- **`model_statistical_report.txt`** - Comprehensive statistical analysis report
- **`model_statistical_results.csv`** - Detailed statistical results table

## Features Included

Each session is transformed into 982 features:

### Per-Trial Features (7 × 140 trials = 980 features)
- `trial{N}_dot_latency_pog` - Dot latency for trial N
- `trial{N}_cue_latency_pog` - Cue latency for trial N
- `trial{N}_trial_saccade_data_quality_pog` - Saccade quality (0=good, 1=bad)
- `trial{N}_cue_latency_pog_good` - Cue latency quality indicator
- `trial{N}_percent_bottom_freeface_pog` - Percent fixation on bottom face
- `trial{N}_percent_top_freeface_pog` - Percent fixation on top face
- `trial{N}_fixation_quality` - Overall fixation quality

### Session-Level Features (2 features)
- `percent_loss_late` - Percent data loss in session
- `session_saccade_data_quality_pog` - Overall session quality

### Target Variables (added from PCL scores)
- `pcl_score` - Raw PCL score (continuous)
- `ptsd` - PTSD status (Positive/Negative)
- `ptsd_bin` - Binary PTSD (1=Positive, 0=Negative)

## Troubleshooting

### AWS Access Issues
- Run `python test_aws_connection.py` to diagnose
- Check AWS credentials: `aws configure list`
- Verify IAM permissions (see AWS_SETUP.md)

### Missing PCL Scores
- Some sessions may not have PCL scores
- Check merge statistics in fetch_pcl_scores.py output
- These sessions will have NaN in PCL columns

### Memory Issues
- Process files in smaller batches if needed
- Reduce number of parallel downloads in download_s3_sessions.py

## Advanced Usage

### Complete YAML Configuration Reference

All settings can be controlled via YAML files. See `configs/six_variations.yml` for the main example.

#### Multi-Target Analysis
```yaml
multi_target:
  - target: ptsd_bin
    variations:
      kfold: true
      kfold_filtered: true
  - target: pcl_score
    variations:
      kfold: true
```

#### Feature Engineering
```yaml
feature_engineering:
  use_rfecv: true         # Recursive Feature Elimination with CV
  rfecv_step: 1          # Features to remove per iteration
  rfecv_min_features: 5  # Minimum features to keep
```

#### Model Selection
```yaml
models:
  # Option 1: Include specific models
  include_models:
    - 'XGBoost'
    - 'Deep NN'
    - 'Random Forest'
  
  # Option 2: Exclude models (if include_models is empty)
  exclude_models:
    - 'Naive Bayes'
```

#### Hyperparameter Tuning
```yaml
hyperparameter_tuning:
  enabled: true           # Enable tuning
  use_random_search: true # RandomizedSearchCV (faster)
  n_iter: 50             # Search iterations
  top_models: 3          # Tune top N models
  models_to_tune:        # Or specify models
    - 'XGBoost'
    - 'Random Forest'
```

#### Automated Pipeline
```yaml
automated_pipeline:
  enabled: true          # Run RFECV + tuning automatically
```

### Example Configuration Files

#### Quick Test Run (configs/quick_test.yml)
```yaml
sample_fraction: 0.05    # 5% of data
n_iterations: 1         # Single k-fold
models:
  include_models: ['Random Forest', 'XGBoost']
hyperparameter_tuning:
  enabled: false        # Skip for speed
```

#### Full Analysis (configs/full_analysis.yml)
```yaml
sample_fraction: 1.0    # All data
n_iterations: 10        # 10x k-fold
automated_pipeline:
  enabled: true         # Full automation
models:
  exclude_models: []    # Use all 11 models
```

#### Deep Learning Focus (configs/deep_learning.yml)
```yaml
models:
  include_models:
    - 'Neural Network'
    - 'Deep NN'
    - 'Wide NN'
    - 'Funnel NN'
hyperparameter_tuning:
  enabled: true
  models_to_tune: ['Deep NN', 'Funnel NN']
```

### Command Line Options

#### Remove Intermediate PCL Scores
```bash
python fetch_pcl_scores.py --remove-intermediate --pcl-threshold 33 --pcl-buffer 8
```

#### Individual Pipeline Runs
```bash
# Automated pipeline (RFECV + tuning)
python ml_pipeline_kfold.py --target ptsd_bin --auto

# Just RFECV
python ml_pipeline_with_pcl.py --target pcl_score --rfecv

# Just hyperparameter tuning
python ml_pipeline.py --target ptsd_bin --tune

# Custom parameters
python ml_pipeline_with_pcl.py --test-size 0.3 --sample-fraction 0.1
```

## Output Structure

```
ml_output/
├── run_XX_YYYYMMDD_HHMMSS/
│   ├── model_comparison.png
│   ├── confusion_matrices.png (classification)
│   ├── regression_plots.png (regression)
│   ├── feature_importance_*.png
│   ├── roc_curves.png (classification)
│   ├── model_results_summary.csv
│   ├── kfold_results_summary.csv (k-fold runs)
│   ├── detailed_predictions.csv
│   ├── run_metadata.json
│   └── report.html
└── combined_report.html (multi-run analysis)
```

## Performance Tips

1. **Start small**: Use `sample_fraction: 0.05` for testing
2. **Focus models**: Use `include_models` to test specific models
3. **Parallel processing**: Set `n_jobs: -1` (default) for all cores
4. **Memory issues**: Reduce models or sample size

## Contact

For issues or questions, please check the error messages and AWS logs first, then verify your AWS permissions match those listed in AWS_SETUP.md.
