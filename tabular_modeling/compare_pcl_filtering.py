#!/usr/bin/env python3
"""
Compare ML pipeline results with and without intermediate PCL filtering
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(output_dir):
    """Load results from a pipeline run"""
    metadata_file = output_dir / 'kfold_metadata.json'
    if not metadata_file.exists():
        metadata_file = output_dir / 'run_metadata.json'
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def compare_filtering_results():
    """Compare results with and without PCL filtering"""
    # Find runs with and without filtering
    ml_output = Path('ml_output')
    
    filtered_runs = []
    unfiltered_runs = []
    
    for run_dir in ml_output.iterdir():
        if run_dir.is_dir() and 'ptsd_bin' in run_dir.name and 'kfold' in run_dir.name:
            try:
                metadata = load_results(run_dir)
                if 'pcl_filtering' in metadata:
                    if metadata['pcl_filtering']['remove_intermediate']:
                        filtered_runs.append((run_dir.name, metadata))
                    else:
                        unfiltered_runs.append((run_dir.name, metadata))
                else:
                    # Runs without pcl_filtering key are unfiltered
                    unfiltered_runs.append((run_dir.name, metadata))
            except:
                continue
    
    print("PCL Filtering Comparison")
    print("=" * 60)
    
    # Compare sample sizes
    print("\nSample Sizes:")
    for name, meta in unfiltered_runs[-1:]:
        print(f"  Without filtering: {meta['data_info']['total_samples']} samples")
    for name, meta in filtered_runs[-1:]:
        print(f"  With filtering: {meta['data_info']['total_samples']} samples")
        print(f"  Excluded range: PCL scores {meta['pcl_filtering']['excluded_range']}")
    
    # Compare performance metrics
    if filtered_runs and unfiltered_runs:
        print("\nPerformance Comparison (Most Recent Runs):")
        
        # Get most recent of each type
        _, filtered_meta = filtered_runs[-1]
        _, unfiltered_meta = unfiltered_runs[-1]
        
        if 'model_results_summary' in filtered_meta and 'model_results_summary' in unfiltered_meta:
            for model in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
                if model in filtered_meta['model_results_summary'] and model in unfiltered_meta['model_results_summary']:
                    print(f"\n  {model}:")
                    
                    # Get AUC scores
                    if 'auc_mean' in filtered_meta['model_results_summary'][model]:
                        filtered_auc = filtered_meta['model_results_summary'][model]['auc_mean']
                        unfiltered_auc = unfiltered_meta['model_results_summary'][model]['auc_mean']
                        
                        print(f"    AUC without filtering: {unfiltered_auc:.4f}")
                        print(f"    AUC with filtering:    {filtered_auc:.4f}")
                        print(f"    Difference:            {filtered_auc - unfiltered_auc:+.4f}")
                    elif 'test_roc_auc' in filtered_meta.get('model_results', {}).get(model, {}):
                        # For regular runs
                        filtered_auc = filtered_meta['model_results'][model]['test_roc_auc']
                        unfiltered_auc = unfiltered_meta['model_results'][model]['test_roc_auc']
                        
                        print(f"    AUC without filtering: {unfiltered_auc:.4f}")
                        print(f"    AUC with filtering:    {filtered_auc:.4f}")
                        print(f"    Difference:            {filtered_auc - unfiltered_auc:+.4f}")
    
    print("\n" + "=" * 60)
    print("\nSummary:")
    print("Removing intermediate PCL scores (25-41) reduces the dataset size")
    print("but may improve model performance by creating clearer separation")
    print("between PTSD positive and negative cases.")

if __name__ == "__main__":
    compare_filtering_results()