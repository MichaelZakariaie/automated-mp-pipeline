#!/usr/bin/env python3
"""
Fetch PCL scores from AWS Athena and merge with processed session data
"""

import pandas as pd
import awswrangler as wr
from pathlib import Path
import argparse
from tqdm import tqdm
import boto3
import sys

def verify_aws_access():
    """Verify AWS credentials and access"""
    try:
        # Test AWS credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        print(f"AWS User/Role: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"Error: Unable to access AWS: {e}")
        print("\nPlease configure AWS credentials:")
        print("  1. Run: aws configure")
        print("  2. Or set environment variables:")
        print("     export AWS_ACCESS_KEY_ID=your_key")
        print("     export AWS_SECRET_ACCESS_KEY=your_secret")
        print("     export AWS_DEFAULT_REGION=us-east-1")
        return False

def fetch_pcl_scores():
    """Fetch PCL scores from AWS Athena"""
    print("Fetching PCL scores from AWS Athena...")
    
    QUERY = "SELECT session_id, pcl_score, ptsd FROM data_quality.mp_pcl_scores"
    DATABASE = "data_quality"
    
    try:
        # Read from Athena using AWS Data Wrangler
        df_chunks = wr.athena.read_sql_query(
            QUERY, 
            database=DATABASE, 
            ctas_approach=True, 
            chunksize=True
        )
        
        # Concatenate all chunks
        df_pcl = pd.concat(df_chunks, axis=0)
        
        print(f"Fetched {len(df_pcl)} PCL scores")
        print(f"Unique sessions with PCL scores: {df_pcl['session_id'].nunique()}")
        
        # Extract UUID part from session_id (remove timestamp suffix)
        df_pcl['session_id_uuid'] = df_pcl['session_id'].str.split('_').str[0]
        
        # Create binary PTSD column
        df_pcl['ptsd_bin'] = df_pcl['ptsd'].map({'Negative': 0, 'Positive': 1})
        
        # Show distribution
        print("\nPTSD distribution:")
        print(df_pcl['ptsd'].value_counts())
        
        # Keep original session_id for reference but use UUID for merging
        df_pcl['session_id_full'] = df_pcl['session_id']
        df_pcl['session_id'] = df_pcl['session_id_uuid']
        
        return df_pcl
        
    except Exception as e:
        print(f"Error fetching PCL scores: {e}")
        print("Make sure you have AWS credentials configured and awswrangler installed.")
        return None

def merge_pcl_with_sessions(session_data_path, pcl_df, output_dir='processed_sessions'):
    """Merge PCL scores with processed session data"""
    
    print(f"\nLoading session data from {session_data_path}...")
    
    # Load processed session data
    if Path(session_data_path).suffix == '.parquet':
        df_sessions = pd.read_parquet(session_data_path)
    else:
        df_sessions = pd.read_csv(session_data_path)
    
    print(f"Session data shape: {df_sessions.shape}")
    print(f"Unique sessions in data: {df_sessions['session_id'].nunique()}")
    
    # Merge PCL scores with session data
    print("\nMerging PCL scores with session data...")
    
    # Keep only necessary PCL columns
    pcl_columns = ['session_id', 'pcl_score', 'ptsd', 'ptsd_bin']
    df_pcl_subset = pcl_df[pcl_columns].drop_duplicates(subset=['session_id'])
    
    # Merge
    df_merged = df_sessions.merge(
        df_pcl_subset, 
        on='session_id', 
        how='left'
    )
    
    # Report merge statistics
    n_matched = df_merged['pcl_score'].notna().sum()
    n_unmatched = df_merged['pcl_score'].isna().sum()
    
    print(f"\nMerge results:")
    print(f"  - Sessions with PCL scores: {n_matched}")
    print(f"  - Sessions without PCL scores: {n_unmatched}")
    
    if n_matched > 0:
        print(f"\nPTSD distribution in merged data:")
        print(df_merged['ptsd_bin'].value_counts(dropna=False))
    
    # Save merged data
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    output_parquet = output_path / 'all_sessions_with_pcl.parquet'
    output_csv = output_path / 'all_sessions_with_pcl.csv'
    
    df_merged.to_parquet(output_parquet, index=False)
    df_merged.to_csv(output_csv, index=False)
    
    print(f"\nSaved merged data to:")
    print(f"  - {output_parquet}")
    print(f"  - {output_csv}")
    
    return df_merged

def remove_intermediate_pcl(df, ptsd_threshold=33, buffer=8):
    """
    Remove people who scored near the PTSD threshold on PCL
    This can improve model performance by removing ambiguous cases
    
    Args:
        df: DataFrame with pcl_score column
        ptsd_threshold: PCL score threshold for PTSD (default: 33)
        buffer: Buffer around threshold to exclude (default: 8)
        
    Returns:
        DataFrame with intermediate scores removed
    """
    print(f"\nRemoving intermediate PCL scores (threshold: {ptsd_threshold}, buffer: ±{buffer})...")
    
    initial_count = len(df)
    
    # Remove scores between (threshold - buffer) and (threshold + buffer)
    df_filtered = df[
        ~df['pcl_score'].between(
            ptsd_threshold - buffer, 
            ptsd_threshold + buffer, 
            inclusive='both'
        )
    ].copy()
    
    removed_count = initial_count - len(df_filtered)
    
    print(f"Removed {removed_count} sessions with intermediate PCL scores")
    print(f"Remaining sessions: {len(df_filtered)}")
    
    if 'ptsd_bin' in df_filtered.columns:
        print("\nPTSD distribution after filtering:")
        print(df_filtered['ptsd_bin'].value_counts(dropna=False))
    
    return df_filtered

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fetch PCL scores and merge with session data')
    parser.add_argument('--session-data', default='processed_sessions/all_sessions_processed.parquet',
                        help='Path to processed session data')
    parser.add_argument('--output-dir', default='processed_sessions',
                        help='Directory to save output files')
    parser.add_argument('--remove-intermediate', action='store_true',
                        help='Remove intermediate PCL scores near threshold')
    parser.add_argument('--pcl-threshold', type=int, default=33,
                        help='PCL threshold for PTSD (default: 33)')
    parser.add_argument('--pcl-buffer', type=int, default=8,
                        help='Buffer around PCL threshold (default: 8)')
    
    args = parser.parse_args()
    
    # Verify AWS access first
    if not verify_aws_access():
        sys.exit(1)
    
    # Check if session data exists
    if not Path(args.session_data).exists():
        print(f"Error: Session data not found at {args.session_data}")
        print("Please run process_all_sessions.py first.")
        return
    
    # Fetch PCL scores
    df_pcl = fetch_pcl_scores()
    
    if df_pcl is None:
        print("Failed to fetch PCL scores. Exiting.")
        return
    
    # Optionally remove intermediate PCL scores
    if args.remove_intermediate:
        df_pcl = remove_intermediate_pcl(
            df_pcl, 
            ptsd_threshold=args.pcl_threshold,
            buffer=args.pcl_buffer
        )
    
    # Merge with session data
    df_merged = merge_pcl_with_sessions(
        args.session_data, 
        df_pcl, 
        args.output_dir
    )
    
    print("\nDone! PCL scores have been merged with session data.")

if __name__ == "__main__":
    main()