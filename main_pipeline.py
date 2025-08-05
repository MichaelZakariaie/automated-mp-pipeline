#!/usr/bin/env python3
"""
Main orchestration script for the Automated MP Pipeline
Coordinates between computer vision, time series analysis (yochlol), and tabular modeling
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

# Add subdirectories to path
sys.path.append(str(Path(__file__).parent / 'yochlol'))
sys.path.append(str(Path(__file__).parent / 'tabular_modeling'))

# Import modules from subprojects
from yochlol import get_data, wrangle_ts, train
from tabular_modeling import (
    download_s3_sessions, 
    process_all_sessions, 
    fetch_pcl_scores,
    ml_pipeline_with_pcl
)

# Import our custom modules
from cv_data_generator import ComputerVisionDataGenerator
from cohort_manager import CohortConfigManager


class AutomatedMPPipeline:
    """Main pipeline orchestrator for MP analysis"""
    
    def __init__(self, config_path='pipeline_config.yaml'):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.cohort_manager = CohortConfigManager(config_path)
        self.setup_logging()
        self.setup_directories()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config['execution']['log_level'])
        log_file = self.config['execution']['log_file']
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Pipeline initialized")
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['pipeline']['output_dir'],
            self.config['pipeline']['temp_dir'],
            self.config['pipeline']['reports_dir']
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def run_stage_1_time_series(self):
        """
        Stage 1: Generate or fetch time series data
        This is where computer vision would run on videos or we pull from AWS
        """
        self.logger.info("Starting Stage 1: Time Series Data Generation/Acquisition")
        
        mode = self.config['pipeline']['mode']
        
        if mode == 'local_cv':
            self.logger.info("Running in local computer vision mode")
            self._run_computer_vision()
        elif mode == 'aws':
            self.logger.info("Running in AWS data pull mode")
            self._pull_aws_time_series()
        else:
            raise ValueError(f"Unknown pipeline mode: {mode}")
            
        self.logger.info("Stage 1 completed successfully")
        
    def _run_computer_vision(self):
        """Run computer vision algorithms on videos (or generate dummy data)"""
        cv_config = self.config['computer_vision']
        
        if cv_config['use_dummy_data']:
            self.logger.info("Generating dummy CV data for testing")
            
            generator = ComputerVisionDataGenerator(
                output_dir=Path(self.config['pipeline']['temp_dir']) / 'cv_output',
                sessions_count=cv_config['dummy_sessions_count'],
                trials_per_session=cv_config['dummy_trials_per_session'],
                fps=cv_config['fps']
            )
            
            generator.generate_all_sessions()
            self.logger.info("Dummy CV data generation completed")
        else:
            # Future: Real CV processing
            self.logger.error("Real CV processing not yet implemented")
            raise NotImplementedError("Real CV processing will be implemented later")
            
    def _pull_aws_time_series(self):
        """Pull time series data from AWS using yochlol approach"""
        self.logger.info("Pulling time series data from AWS")
        
        # Get cohort-specific environment variables
        yochlol_env = os.environ.copy()
        yochlol_env.update(self.cohort_manager.get_yochlol_env_vars())
        
        # Log cohort information
        self.logger.info(f"Using cohort: {self.cohort_manager.active_cohort}")
        self.logger.info(f"S3 bucket: {yochlol_env['BUCKET_NAME']}")
        self.logger.info(f"S3 prefix: {yochlol_env['S3_PREFIX']}")
        
        # Change to yochlol directory for relative paths
        original_cwd = os.getcwd()
        os.chdir(Path(__file__).parent / 'yochlol')
        
        try:
            # Configure AWS
            subprocess.run(['bash', 'aws_throughput_config.sh'], check=True)
            
            # Download compliance videos if needed
            if self.config['yochlol']['compliance_metrics']:
                self.logger.info("Downloading compliance videos")
                subprocess.run([
                    sys.executable, 'download_compliance_vids.py'
                ], env=yochlol_env, check=True)
                
                # Run compliance analysis
                self.logger.info("Running compliance analysis")
                subprocess.run([
                    sys.executable, 'run_compliance.py',
                    '--limit', '10'  # Limit for testing
                ], env=yochlol_env, check=True)
                
                # Get compliance stats
                subprocess.run([
                    sys.executable, 'get_stats.py'
                ], env=yochlol_env, check=True)
            
            # Get PTSD data from AWS
            self.logger.info("Fetching PTSD data from AWS")
            subprocess.run([
                sys.executable, 'get_data.py', '--save'
            ], env=yochlol_env, check=True)
            
        finally:
            os.chdir(original_cwd)
            
    def run_stage_2_time_series_analysis(self):
        """
        Stage 2: Run time series analysis using yochlol
        """
        self.logger.info("Starting Stage 2: Time Series Analysis")
        
        # Change to yochlol directory
        original_cwd = os.getcwd()
        os.chdir(Path(__file__).parent / 'yochlol')
        
        try:
            # Wrangle time series data
            if self.config['yochlol']['wrangle_time_series']:
                self.logger.info("Wrangling time series data")
                
                # Find the most recent data file
                data_files = list(Path('.').glob('*face_pairs*.parquet'))
                if not data_files:
                    raise FileNotFoundError("No time series data files found")
                
                latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
                
                subprocess.run([
                    sys.executable, 'wrangle_ts.py',
                    '--data-path', str(latest_data),
                    '--task', self.config['yochlol']['task'],
                    '--save'
                ], check=True)
            
            # Run rocket models
            self.logger.info("Training time series models")
            
            # Find wrangled data
            wrangled_files = list(Path('.').glob('wrangled_ts_*.parquet'))
            if not wrangled_files:
                raise FileNotFoundError("No wrangled time series files found")
                
            latest_wrangled = max(wrangled_files, key=lambda x: x.stat().st_mtime)
            
            subprocess.run([
                sys.executable, 'train.py',
                '--data-path', str(latest_wrangled),
                '--rkt-chunksize', str(self.config['yochlol']['rocket_chunksize'])
            ], check=True)
            
            # Copy results to output directory
            results_dir = Path(self.config['pipeline']['output_dir']) / 'time_series_results'
            results_dir.mkdir(exist_ok=True)
            
            for result_file in Path('.').glob('*.pkl'):
                shutil.copy2(result_file, results_dir)
                
        finally:
            os.chdir(original_cwd)
            
        self.logger.info("Stage 2 completed successfully")
        
    def wait_for_tabular_data(self):
        """
        Wait for external team to process data and make tabular data available
        In practice, this might check AWS periodically or wait for a signal
        """
        self.logger.info("Waiting for tabular data to be available...")
        
        # For now, just check if tabular data exists
        # In production, this would poll AWS or wait for notification
        
        input("Press Enter when tabular data is ready in AWS...")
        
    def run_stage_3_tabular_analysis(self):
        """
        Stage 3: Run tabular modeling analysis
        """
        self.logger.info("Starting Stage 3: Tabular Analysis")
        
        # Get cohort-specific environment variables
        tabular_env = os.environ.copy()
        tabular_env.update(self.cohort_manager.get_tabular_env_vars())
        
        # Log cohort information
        self.logger.info(f"Using cohort: {self.cohort_manager.active_cohort}")
        self.logger.info(f"S3 bucket: {tabular_env['S3_BUCKET']}")
        self.logger.info(f"S3 prefix: {tabular_env['S3_PREFIX']}")
        
        # Change to tabular_modeling directory
        original_cwd = os.getcwd()
        os.chdir(Path(__file__).parent / 'tabular_modeling')
        
        try:
            # Download session data from S3
            self.logger.info("Downloading session data from S3")
            subprocess.run([
                sys.executable, 'download_s3_sessions.py'
            ], env=tabular_env, check=True)
            
            # Process all sessions
            self.logger.info("Processing session data")
            subprocess.run([
                sys.executable, 'process_all_sessions.py'
            ], check=True)
            
            # Fetch PCL scores
            self.logger.info("Fetching PCL scores")
            subprocess.run([
                sys.executable, 'fetch_pcl_scores.py'
            ], check=True)
            
            # Run ML pipeline for each target
            for target in self.config['tabular_modeling']['targets']:
                self.logger.info(f"Running ML pipeline for target: {target}")
                
                cmd = [
                    sys.executable, 'ml_pipeline_with_pcl.py',
                    '--target', target,
                    '--sample-fraction', str(self.config['tabular_modeling']['sample_fraction']),
                    '--test-size', str(self.config['tabular_modeling']['test_size'])
                ]
                
                if self.config['tabular_modeling']['use_rfecv']:
                    cmd.append('--rfecv')
                    
                if self.config['tabular_modeling']['hyperparameter_tuning']:
                    cmd.append('--tune')
                    
                subprocess.run(cmd, check=True)
            
            # Copy results to output directory
            results_dir = Path(self.config['pipeline']['output_dir']) / 'tabular_results'
            results_dir.mkdir(exist_ok=True)
            
            # Copy ML output
            if Path('ml_output').exists():
                shutil.copytree('ml_output', results_dir / 'ml_output', dirs_exist_ok=True)
                
        finally:
            os.chdir(original_cwd)
            
        self.logger.info("Stage 3 completed successfully")
        
    def generate_final_report(self):
        """Generate comprehensive final report"""
        self.logger.info("Generating final report")
        
        # TODO: Implement comprehensive report generation
        # For now, just copy existing reports
        
        report_dir = Path(self.config['pipeline']['reports_dir'])
        
        # Copy time series results
        ts_results = Path(self.config['pipeline']['output_dir']) / 'time_series_results'
        if ts_results.exists():
            shutil.copytree(ts_results, report_dir / 'time_series', dirs_exist_ok=True)
            
        # Copy tabular results
        tab_results = Path(self.config['pipeline']['output_dir']) / 'tabular_results'
        if tab_results.exists():
            shutil.copytree(tab_results, report_dir / 'tabular', dirs_exist_ok=True)
            
        self.logger.info("Report generation completed")
        
    def run_full_pipeline(self):
        """Run the complete pipeline end-to-end"""
        self.logger.info("Starting full pipeline execution")
        
        try:
            # Stage 1: Generate/fetch time series data
            self.run_stage_1_time_series()
            
            # Stage 2: Time series analysis
            self.run_stage_2_time_series_analysis()
            
            # Wait for tabular data
            self.wait_for_tabular_data()
            
            # Stage 3: Tabular analysis
            self.run_stage_3_tabular_analysis()
            
            # Generate final report
            self.generate_final_report()
            
            self.logger.info("Pipeline execution completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Automated MP Pipeline')
    parser.add_argument('--config', default='pipeline_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--stage', choices=['time_series', 'tabular', 'full'],
                        default='full',
                        help='Which stage to run')
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = AutomatedMPPipeline(args.config)
    
    # Run requested stage
    if args.stage == 'time_series':
        pipeline.run_stage_1_time_series()
        pipeline.run_stage_2_time_series_analysis()
    elif args.stage == 'tabular':
        pipeline.run_stage_3_tabular_analysis()
    else:  # full
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()