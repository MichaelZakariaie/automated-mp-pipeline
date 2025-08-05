# Cohort Configuration Requirements

This document outlines what information is needed to configure the pipeline for a new cohort.

## Required Information for New Cohorts

When setting up the pipeline for a new cohort, you'll need the following information:

### 1. Basic Cohort Information
- **Cohort ID**: Numeric identifier (e.g., 6, 7, 8)
- **Description**: Brief description (e.g., "Spring 2024 Veterans Study")

### 2. AWS S3 Locations

#### For Time Series Data (yochlol)
- **Compliance Videos Bucket**: Where raw videos are stored
  - Example: `senseye-ptsd`
- **Compliance Videos Prefix**: Path within bucket
  - Example: `public/ptsd_ios/cohort_6/`

#### For Tabular Data (tabular modeling)
- **Data Bucket**: Where processed session data is stored
  - Example: `senseye-data-quality`
- **Data Prefix**: Path to session files
  - Example: `cohort_6_saturn_uploads/`

#### For Computer Vision Output (when running locally)
- **Upload Bucket**: Where to upload generated time series
- **Upload Prefix Template**: Path pattern with placeholders
  - Example: `cohort_6_uploads/{session_id}/face_pairs/processed_v3/`

### 3. AWS Athena Configuration

- **Database Name**: Usually `data_quality`
- **Table Names**:
  - **PCL Scores Table**: Table containing PTSD assessment scores
    - Example: `mp_pcl_scores` or `mp_pcl_scores_v2`
  - **Surveys Table**: Qualtrics survey data
    - Example: `qualtrics_surveys_unified`
  - **RAPS Scores Table**: Additional assessment scores
    - Example: `mp_raps_scores`
  - **Session Completion Table**: Tracking table
    - Example: `master_query_session_completion_check`
  - **App Session Details Table**: Session metadata
    - Example: `messy_prototyping_app_session_details`

### 4. Task Configuration

- **Available Tasks**: List of all tasks this cohort might have
  - Example: `["face_pairs", "calibration_1", "calibration_2", "plr", "mckinnon"]`
- **Default Tasks**: Which tasks to process by default
  - Example: `["face_pairs"]`
- **Video Types**: Which video types to download for compliance
  - Example: `["calibration_1", "plr", "face_pairs", "mckinnon"]`
- **Sample Counts**: Number of trials/samples per task
  - Example:
    ```yaml
    face_pairs: 140
    calibration_1: 9
    calibration_2: 9
    plr: 90
    mckinnon: 30
    ```

### 5. File Naming Patterns

- **Session File Pattern**: How individual session files are named
  - Example: `{session_id}_inter_pir_face_pairs_v2_latedwell_pog.parquet`
  - Placeholders: `{session_id}`, `{timestamp}`, `{cohort_id}`
  
- **Time Series Pattern**: Output format for time series data
  - Example: `MP_cohort_{cohort_id}_{num_sessions}sessions_{timestamp}_{suffix}.parquet`
  
- **Wrangled Pattern**: Format for processed time series
  - Example: `wrangled_ts_{task}_{sessions}sessions_{fps}fps_{timestamp}.parquet`

### 6. Machine Learning Targets

- **Classification Target**: Binary PTSD prediction column
  - Example: `ptsd_bin`
- **Regression Target**: Continuous score column
  - Example: `pcl_score`
- **Alternative Targets**: Other available prediction targets
  - Example: `["raps_binary_ptsd", "ptsd_severity"]`

### 7. Processing Parameters (Optional)

- **Download Workers**: Parallel download threads
  - Default: 24
- **Download Buffer**: Queue size for downloads
  - Default: 20
- **Max Parallel Sessions**: Concurrent processing limit
  - Default: 10

## Example Configuration Entry

Here's a complete example for adding "cohort_7":

```yaml
cohort_7:
  cohort_id: 7
  description: "Cohort 7 - Fall 2024 Veterans Study"
  
  s3:
    compliance_bucket: "senseye-ptsd"
    compliance_prefix: "public/ptsd_ios/cohort_7/"
    data_bucket: "senseye-data-quality" 
    data_prefix: "cohort_7_saturn_uploads/"
    upload_bucket: "senseye-data-quality"
    upload_prefix: "cohort_7_uploads/{session_id}/face_pairs/processed_v3/"
  
  athena:
    database: "data_quality"
    tables:
      surveys: "qualtrics_surveys_unified"
      pcl_scores: "mp_pcl_scores_cohort7"
      raps_scores: "mp_raps_scores"
      session_completion: "cohort7_session_completion"
      app_session_details: "cohort7_app_details"
  
  tasks:
    available: ["face_pairs", "calibration_1", "calibration_2", "plr", "mckinnon", "emotion_task"]
    default: ["face_pairs", "emotion_task"]
    video_types: ["calibration_1", "plr", "face_pairs", "mckinnon", "emotion_task"]
    sample_counts:
      face_pairs: 140
      calibration_1: 9
      calibration_2: 9
      plr: 90
      mckinnon: 30
      emotion_task: 120
  
  file_patterns:
    session_file_pattern: "{session_id}_cohort7_face_pairs_processed.parquet"
    time_series_pattern: "C7_{cohort_id}_{num_sessions}sessions_{timestamp}.parquet"
    wrangled_pattern: "wrangled_c7_{task}_{sessions}sessions_{fps}fps_{timestamp}.parquet"
  
  ml_targets:
    classification: "ptsd_bin"
    regression: "pcl_score"
    alternatives:
      - "severity_score"
      - "treatment_response"
```

## How to Add a New Cohort

1. Gather all required information listed above
2. Edit `cohort_config.yaml`
3. Add your cohort configuration following the template
4. Test with: `python cohort_manager.py --cohort your_cohort --summary`
5. Run pipeline: `python run_pipeline.py time_series --cohort your_cohort`

## Verifying Configuration

Before running the pipeline with a new cohort:

1. **Check S3 Access**:
   ```bash
   aws s3 ls s3://your-bucket/your-prefix/
   ```

2. **Verify Athena Tables**:
   ```bash
   aws athena list-table-metadata --database-name your_database
   ```

3. **Test Configuration**:
   ```bash
   python cohort_manager.py --cohort your_cohort --summary
   ```

4. **Dry Run Update**:
   ```bash
   python update_cohort_scripts.py --cohort your_cohort --dry-run
   ```