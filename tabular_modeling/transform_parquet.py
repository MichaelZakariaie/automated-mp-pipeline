import pandas as pd
import numpy as np
from pathlib import Path

def transform_df_to_single_row(df):
    """
    Transform dataframe into a single row with columns named trial[X]_[feature]
    where X is the trial number and feature is one of the specified columns.
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
    # Note: We are EXCLUDING: num_unique_sacc, dot_latency_pog_good
    
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
                        new_data[col_name] = trial_row.iloc[0][col]
                    else:
                        new_data[col_name] = np.nan  # Missing trial
    
    # Add single columns (take first occurrence or aggregate as needed)
    for col in single_columns:
        if col in df.columns:
            new_data[col] = df[col].iloc[0]  # Take first value
    
    # Create single-row dataframe
    result_df = pd.DataFrame([new_data])
    
    return result_df

def main():
    # Load the parquet file
    parquet_file = "analytical_sessions/1e0e9259-1ae8-4e9d-bc87-0800ed4c3015_1752077526035_inter_pir_face_pairs_v2_latedwell_pog.parquet"
    df = pd.read_parquet(parquet_file)
    
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Number of unique trials in data: {df['trial'].nunique()}")
    print(f"Trial range: {df['trial'].min()} to {df['trial'].max()}")
    print(f"Missing trials: {set(range(140)) - set(df['trial'].values)}")
    
    # Check columns
    print("\nChecking which columns exist:")
    columns_to_check = [
        'dot_latency_pog', 'cue_latency_pog', 'num_unique_sacc',
        'trial_saccade_data_quality_pog', 'cue_latency_pog_good',
        'dot_latency_pog_good', 'percent_bottom_freeface_pog',
        'percent_top_freeface_pog', 'fixation_quality',
        'percent_loss_late', 'session_saccade_data_quality_pog'
    ]
    
    for col in columns_to_check:
        print(f"  {col}: {'EXISTS' if col in df.columns else 'MISSING'}")
    
    # Transform
    transformed_df = transform_df_to_single_row(df)
    
    print(f"\nTransformed DataFrame shape: {transformed_df.shape}")
    print(f"Expected shape: (1, {7 * 140 + 2})")  # 7 trial columns × 140 trials + 2 single columns = 982
    
    # Analyze the columns
    trial_cols = [col for col in transformed_df.columns if col.startswith('trial')]
    single_cols = [col for col in transformed_df.columns if not col.startswith('trial')]
    
    print(f"\nColumn breakdown:")
    print(f"  Trial columns: {len(trial_cols)}")
    print(f"  Single columns: {len(single_cols)}")
    print(f"  Single column names: {single_cols}")
    
    # Count columns by feature type
    feature_counts = {}
    for col in trial_cols:
        parts = col.split('_', 1)
        if len(parts) > 1:
            feature = parts[1]
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    print(f"\nTrial columns by feature:")
    for feature, count in sorted(feature_counts.items()):
        print(f"  {feature}: {count} trials")
    
    # Sample of missing data
    print(f"\nSample values for missing trials:")
    missing_trial_cols = [col for col in transformed_df.columns if 'trial13_' in col or 'trial41_' in col]
    for col in missing_trial_cols[:3]:
        print(f"  {col}: {transformed_df[col].iloc[0]}")

if __name__ == "__main__":
    main()