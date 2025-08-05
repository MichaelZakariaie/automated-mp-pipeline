#!/bin/bash

# Full pipeline script to download, process, and analyze session data

echo "=========================================="
echo "Session Data Analysis Pipeline with PCL"
echo "=========================================="

# Step 1: Download data from S3
echo -e "\n[Step 1/5] Downloading session files from S3..."
python download_s3_sessions.py

# Check if download was successful
if [ ! -d "downloaded_sessions" ] || [ -z "$(ls -A downloaded_sessions/*.parquet 2>/dev/null)" ]; then
    echo "Error: No files downloaded. Please check your AWS credentials and try again."
    exit 1
fi

# Step 2: Process the downloaded files
echo -e "\n[Step 2/5] Processing downloaded session files..."
python process_all_sessions.py

# Check if processing was successful
if [ ! -f "processed_sessions/all_sessions_processed.parquet" ]; then
    echo "Error: Processing failed. Please check the error messages above."
    exit 1
fi

# Step 3: Fetch PCL scores and merge with session data
echo -e "\n[Step 3/5] Fetching PCL scores from AWS Athena..."
python fetch_pcl_scores.py

# Check if PCL merge was successful
if [ ! -f "processed_sessions/all_sessions_with_pcl.parquet" ]; then
    echo "Error: PCL score fetching/merging failed. Please check your AWS credentials."
    exit 1
fi

# Step 4: Run ML pipeline for PTSD prediction (binary classification)
echo -e "\n[Step 4/5] Running ML pipeline for PTSD prediction..."
python ml_pipeline_with_pcl.py --target ptsd_bin

# Step 5: Run ML pipeline for PCL score prediction (regression)
echo -e "\n[Step 5/5] Running ML pipeline for PCL score prediction..."
python ml_pipeline_with_pcl.py --target pcl_score

echo -e "\n=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo -e "\nResults saved in:"
echo "  - processed_sessions/: Processed data files"
echo "    - all_sessions_processed.parquet: Session data without PCL"
echo "    - all_sessions_with_pcl.parquet: Session data with PCL scores"
echo "    - all_sessions_with_pcl.csv: Same as above in CSV format"
echo "  - ml_results/: Trained models and results"
echo "    - Models for PTSD binary classification"
echo "    - Models for PCL score regression"
echo "  - Feature importance plots for both targets"
echo "  - Model comparison and ROC curves"