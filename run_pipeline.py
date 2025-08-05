#!/usr/bin/env python3
"""
Simple command-line interface for running the automated MP pipeline
Provides two main commands:
1. run_time_series - Run CV/time series analysis
2. run_tabular - Run tabular analysis after data is ready
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import subprocess
from datetime import datetime


def load_config(config_path='pipeline_config.yaml'):
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_time_series_pipeline(config_path='pipeline_config.yaml'):
    """
    Run the time series portion of the pipeline
    This includes CV processing (or AWS pull) and time series analysis
    """
    print("=" * 60)
    print("AUTOMATED MP PIPELINE - TIME SERIES ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    config = load_config(config_path)
    mode = config['pipeline']['mode']
    
    print(f"Running in {mode.upper()} mode")
    print()
    
    if mode == 'local_cv':
        if config['computer_vision']['use_dummy_data']:
            print("Note: Using dummy CV data for testing")
        else:
            print("Note: Real CV processing not yet implemented")
    
    # Run the pipeline
    cmd = [sys.executable, 'main_pipeline.py', '--config', config_path, '--stage', 'time_series']
    
    print("Starting time series pipeline...")
    print("-" * 60)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("-" * 60)
        print("✓ Time series analysis completed successfully!")
        print()
        print("Next steps:")
        print("1. Wait for the external team to process the data and upload to AWS")
        print("2. Run 'python run_pipeline.py tabular' to complete the analysis")
    else:
        print()
        print("-" * 60)
        print("✗ Time series analysis failed!")
        print("Check the logs for details: pipeline_execution.log")
        return 1
    
    return 0


def run_tabular_pipeline(config_path='pipeline_config.yaml'):
    """
    Run the tabular portion of the pipeline
    This should be run after the external team has processed the data
    """
    print("=" * 60)
    print("AUTOMATED MP PIPELINE - TABULAR ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if time series results exist
    config = load_config(config_path)
    ts_results = Path(config['pipeline']['output_dir']) / 'time_series_results'
    
    if not ts_results.exists():
        print("WARNING: No time series results found!")
        print("Have you run 'python run_pipeline.py time_series' first?")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    print("Starting tabular analysis pipeline...")
    print("This will:")
    print("1. Download tabular data from AWS")
    print("2. Process and merge with PCL scores")
    print("3. Run ML models for PTSD prediction")
    print("4. Generate comprehensive reports")
    print("-" * 60)
    
    # Run the pipeline
    cmd = [sys.executable, 'main_pipeline.py', '--config', config_path, '--stage', 'tabular']
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("-" * 60)
        print("✓ Tabular analysis completed successfully!")
        print()
        print("Results have been saved to:")
        print(f"  - {config['pipeline']['output_dir']}/")
        print(f"  - {config['pipeline']['reports_dir']}/")
        print()
        print("View the reports to see the analysis results.")
    else:
        print()
        print("-" * 60)
        print("✗ Tabular analysis failed!")
        print("Check the logs for details: pipeline_execution.log")
        return 1
    
    return 0


def show_status(config_path='pipeline_config.yaml'):
    """Show current pipeline status"""
    config = load_config(config_path)
    
    print("=" * 60)
    print("AUTOMATED MP PIPELINE - STATUS")
    print("=" * 60)
    print()
    
    # Check configuration
    print(f"Current mode: {config['pipeline']['mode']}")
    print()
    
    # Check output directories
    output_dir = Path(config['pipeline']['output_dir'])
    reports_dir = Path(config['pipeline']['reports_dir'])
    
    print("Output directories:")
    print(f"  Main output: {output_dir} {'[EXISTS]' if output_dir.exists() else '[NOT CREATED]'}")
    print(f"  Reports: {reports_dir} {'[EXISTS]' if reports_dir.exists() else '[NOT CREATED]'}")
    print()
    
    # Check results
    ts_results = output_dir / 'time_series_results'
    tab_results = output_dir / 'tabular_results'
    
    print("Pipeline stages:")
    print(f"  Time series: {'✓ COMPLETED' if ts_results.exists() else '✗ NOT RUN'}")
    print(f"  Tabular: {'✓ COMPLETED' if tab_results.exists() else '✗ NOT RUN'}")
    print()
    
    # Check for specific files
    if ts_results.exists():
        ts_files = list(ts_results.glob('*'))
        print(f"Time series results: {len(ts_files)} files")
        
    if tab_results.exists():
        ml_output = tab_results / 'ml_output'
        if ml_output.exists():
            runs = list(ml_output.glob('run_*'))
            print(f"Tabular ML runs: {len(runs)}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Automated MP Pipeline - Simple CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run time series analysis (first command)
  python run_pipeline.py time_series
  
  # Run tabular analysis (second command, after data is ready)
  python run_pipeline.py tabular
  
  # Check pipeline status
  python run_pipeline.py status
  
  # Use custom config
  python run_pipeline.py time_series --config my_config.yaml
        """
    )
    
    parser.add_argument('command', 
                        choices=['time_series', 'tabular', 'status'],
                        help='Which pipeline stage to run')
    parser.add_argument('--config', 
                        default='pipeline_config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run the requested command
    if args.command == 'time_series':
        return run_time_series_pipeline(args.config)
    elif args.command == 'tabular':
        return run_tabular_pipeline(args.config)
    elif args.command == 'status':
        show_status(args.config)
        return 0


if __name__ == '__main__':
    sys.exit(main())