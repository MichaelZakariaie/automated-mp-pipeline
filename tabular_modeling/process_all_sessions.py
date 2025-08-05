#!/usr/bin/env python3
"""
Process all downloaded session files using the transform_parquet logic
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm

def transform_df_to_single_row(df):
    """
    Transform dataframe into a single row with columns named trial[X]_[feature]
    where X is the trial number and feature is one of the specified columns.
    Creates columns for ALL 140 trials (0-139), with NaN for missing trials.
    """
    # Define the columns we want to transform (per trial) - EXCLUDING num_unique_sacc and dot_latency_pog_good
    trial_columns = [
        'dot_latency_pog', 
        'cue_latency_pog', 
        'trial_saccade_data_quality_pog', 
        'cue_latency_pog_good', 
        'percent_bottom_freeface_pog', 
        'percent_top_freeface_pog',
        'fixation_quality'
    ]
    
    # Define columns that don't have trial numbers (single values)
    single_columns = ['percent_loss_late', 'session_saccade_data_quality_pog']
    
    # Create mapping for trial_saccade_data_quality_pog
    quality_mapping = {val: idx for idx, val in enumerate(df['trial_saccade_data_quality_pog'].unique())}
    
    # Create new dataframe structure
    new_data = {}
    
    # We need to create columns for ALL 140 trials (0-139), not just the ones in the data
    for trial_num in range(140):  # 0 to 139
        for col in trial_columns:
            if col in df.columns:
                # Find if this trial exists in the data
                trial_row = df[df['trial'] == trial_num]
                
                if col == 'trial_saccade_data_quality_pog':
                    # Label encode this column (convert to numbers)
                    col_name = f'trial{trial_num}_{col}'
                    if not trial_row.empty:
                        new_data[col_name] = quality_mapping[trial_row.iloc[0][col]]
                    else:
                        new_data[col_name] = np.nan  # Missing trial
                else:
                    col_name = f'trial{trial_num}_{col}'
                    if not trial_row.empty:
                        value = trial_row.iloc[0][col]
                        # Handle categorical columns with good/bad values
                        if col in ['cue_latency_pog_good', 'fixation_quality']:
                            if value == 'good':
                                new_data[col_name] = 1
                            elif value == 'bad':
                                new_data[col_name] = 0
                            else:
                                new_data[col_name] = np.nan
                        else:
                            new_data[col_name] = value
                    else:
                        new_data[col_name] = np.nan  # Missing trial
    
    # Add single columns (take first occurrence or aggregate as needed)
    for col in single_columns:
        if col in df.columns:
            value = df[col].iloc[0]  # Take first value
            # Handle session_saccade_data_quality_pog encoding
            if col == 'session_saccade_data_quality_pog':
                if value == 'good':
                    new_data[col] = 1
                elif value == 'bad':
                    new_data[col] = 0
                else:
                    new_data[col] = np.nan
            else:
                new_data[col] = value
    
    # Create single-row dataframe
    result_df = pd.DataFrame([new_data])
    
    return result_df

def process_all_sessions(input_dir='downloaded_sessions', output_dir='processed_sessions'):
    """Process all parquet files in the input directory"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all parquet files
    parquet_files = list(Path(input_dir).glob('*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}/")
        return
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Process each file
    all_transformed_dfs = []
    failed_files = []
    
    for parquet_file in tqdm(parquet_files, desc="Processing files"):
        try:
            # Read the parquet file
            df = pd.read_parquet(parquet_file)
            
            # Skip if dataframe is empty
            if df.empty:
                print(f"\nWarning: Empty dataframe in {parquet_file.name}")
                continue
            
            # Transform to single row
            transformed_row = transform_df_to_single_row(df)
            
            # Add metadata
            transformed_row['source_file'] = parquet_file.name
            transformed_row['session_id'] = parquet_file.stem.split('_')[0]
            
            all_transformed_dfs.append(transformed_row)
            
        except Exception as e:
            print(f"\nError processing {parquet_file.name}: {str(e)}")
            failed_files.append((parquet_file.name, str(e)))
    
    if not all_transformed_dfs:
        print("No files were successfully processed!")
        return
    
    # Combine all transformed dataframes
    print("\nCombining all transformed dataframes...")
    combined_df = pd.concat(all_transformed_dfs, ignore_index=True)
    
    # Reorder columns to put identifiers first
    cols = combined_df.columns.tolist()
    id_cols = ['session_id', 'source_file']
    data_cols = [col for col in cols if col not in id_cols]
    combined_df = combined_df[id_cols + data_cols]
    
    print(f"\nFinal combined dataframe shape: {combined_df.shape}")
    print(f"Expected columns per row: {7 * 140 + 2} (7 trial features × 140 trials + 2 single columns)")
    print(f"Actual columns: {len(combined_df.columns) - 2} data columns + 2 identifier columns")
    
    # Save the results
    output_parquet = Path(output_dir) / 'all_sessions_processed.parquet'
    output_csv = Path(output_dir) / 'all_sessions_processed.csv'
    
    combined_df.to_parquet(output_parquet, index=False)
    combined_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Saved to:")
    print(f"  - {output_parquet}")
    print(f"  - {output_csv}")
    
    # Report any failures
    if failed_files:
        print(f"\n✗ Failed to process {len(failed_files)} files:")
        for filename, error in failed_files[:5]:
            print(f"  - {filename}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    # Show summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total sessions processed: {len(combined_df)}")
    print(f"Features per session: {len(data_cols)}")
    
    # Check for missing data
    missing_counts = combined_df.isnull().sum()
    high_missing = missing_counts[missing_counts > len(combined_df) * 0.5]
    if len(high_missing) > 0:
        print(f"\nColumns with >50% missing data: {len(high_missing)}")
        print("Sample columns with high missing rate:")
        for col in list(high_missing.index)[:5]:
            pct = (missing_counts[col] / len(combined_df)) * 100
            print(f"  - {col}: {pct:.1f}% missing")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process all downloaded session files')
    parser.add_argument('--input-dir', default='downloaded_sessions', 
                        help='Directory containing downloaded parquet files')
    parser.add_argument('--output-dir', default='processed_sessions',
                        help='Directory to save processed files')
    
    args = parser.parse_args()
    
    process_all_sessions(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()