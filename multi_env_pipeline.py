#!/usr/bin/env python3
"""
Multi-Environment Pipeline Orchestrator
Runs different parts of the pipeline in their own virtual environments
"""

import os
import sys
import yaml
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Import our cohort manager (this should work in main env)
from cohort_manager import CohortConfigManager


class MultiEnvironmentPipeline:
    """Pipeline orchestrator that uses multiple virtual environments"""
    
    def __init__(self, config_path='pipeline_config.yaml'):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.cohort_manager = CohortConfigManager(config_path)
        self.setup_logging()
        self.setup_directories()
        
        # Environment paths
        self.env_paths = {
            'cv': Path('env_cv'),
            'yochlol': Path('env_yochlol'), 
            'tabular': Path('env_tabular'),
            'main': Path('env_main')
        }
        
        # Verify environments exist
        self._verify_environments()
        
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
        self.logger.info(f"Multi-environment pipeline initialized")
        self.logger.info(f"Active cohort: {self.cohort_manager.active_cohort}")
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['pipeline']['output_dir'],
            self.config['pipeline']['temp_dir'],
            self.config['pipeline']['reports_dir']
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _verify_environments(self):
        """Verify that all required environments exist"""
        missing_envs = []
        for env_name, env_path in self.env_paths.items():
            if not env_path.exists():
                missing_envs.append(env_name)
        
        if missing_envs:
            raise RuntimeError(
                f"Missing virtual environments: {', '.join(missing_envs)}. "
                f"Run './setup_multi_env.sh' first."
            )
    
    def _run_in_environment(self, env_name, script_path, args=None, cwd=None, env_vars=None):
        """Run a script in a specific virtual environment"""
        env_path = self.env_paths[env_name]
        python_exe = env_path / 'bin' / 'python'
        
        if not python_exe.exists():
            raise RuntimeError(f"Python executable not found in {env_name}: {python_exe}")
        
        # Build command
        cmd = [str(python_exe), str(script_path)]
        if args:
            cmd.extend(args)
        
        # Set up environment variables
        subprocess_env = os.environ.copy()
        if env_vars:
            subprocess_env.update(env_vars)
        
        # Add cohort-specific environment variables
        if env_name == 'yochlol':
            subprocess_env.update(self.cohort_manager.get_yochlol_env_vars())
        elif env_name == 'tabular':
            subprocess_env.update(self.cohort_manager.get_tabular_env_vars())
        
        self.logger.info(f"Running in {env_name} environment: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=subprocess_env,
            capture_output=False,  # Let output go to console
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Script failed in {env_name} environment: {script_path}")
        
        return result
    
    def run_computer_vision_stage(self):
        """Stage 1: Run computer vision processing"""
        self.logger.info("Starting Stage 1: Computer Vision Processing")
        
        mode = self.config['pipeline']['mode']
        
        if mode == 'local_cv':
            self.logger.info("Running local computer vision")
            
            # Run CV data generation in CV environment
            cv_script = Path(__file__).parent / 'cv_data_generator.py'
            cv_config = self.config['computer_vision']
            
            args = [
                '--output-dir', str(Path(self.config['pipeline']['temp_dir']) / 'cv_output'),
                '--sessions', str(cv_config['dummy_sessions_count']),
                '--trials', str(cv_config.get('dummy_trials_per_session', 140)),
                '--fps', str(cv_config['fps'])
            ]
            
            self._run_in_environment('cv', cv_script, args)
            
            # TODO: Upload generated data to AWS for external team
            self.logger.info("TODO: Upload CV results to AWS")
            
        elif mode == 'aws':
            self.logger.info("Skipping CV stage - using AWS mode")
        else:
            raise ValueError(f"Unknown pipeline mode: {mode}")
        
        self.logger.info("Stage 1 completed successfully")
    
    def run_time_series_stage(self):
        """Stage 2: Run time series analysis using yochlol"""
        self.logger.info("Starting Stage 2: Time Series Analysis")
        
        # Change to yochlol directory and run scripts
        yochlol_dir = Path(__file__).parent / 'yochlol'
        
        # Configure AWS
        aws_config_script = yochlol_dir / 'aws_throughput_config.sh'
        if aws_config_script.exists():
            self.logger.info("Configuring AWS")
            subprocess.run(['bash', str(aws_config_script)], cwd=yochlol_dir, check=True)
        
        # Download compliance videos if needed
        if self.config['yochlol']['compliance_metrics']:
            self.logger.info("Downloading compliance videos")
            self._run_in_environment(
                'yochlol', 
                yochlol_dir / 'download_compliance_vids.py',
                cwd=yochlol_dir
            )
            
            # Run compliance analysis
            self.logger.info("Running compliance analysis")
            self._run_in_environment(
                'yochlol',
                yochlol_dir / 'run_compliance.py',
                ['--limit', '10'],
                cwd=yochlol_dir
            )
            
            # Get compliance stats
            self._run_in_environment(
                'yochlol',
                yochlol_dir / 'get_stats.py',
                cwd=yochlol_dir
            )
        
        # Get PTSD data from AWS
        self.logger.info("Fetching PTSD data from AWS")
        self._run_in_environment(
            'yochlol',
            yochlol_dir / 'get_data.py',
            ['--save'],
            cwd=yochlol_dir
        )
        
        # Wrangle time series data
        if self.config['yochlol']['wrangle_time_series']:
            self.logger.info("Wrangling time series data")
            
            # Find the most recent data file
            data_files = list(yochlol_dir.glob('*face_pairs*.parquet'))
            if not data_files:
                raise FileNotFoundError("No time series data files found")
            
            latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
            
            self._run_in_environment(
                'yochlol',
                yochlol_dir / 'wrangle_ts.py',
                [
                    '--data-path', str(latest_data),
                    '--task', self.config['yochlol']['task'],
                    '--save'
                ],
                cwd=yochlol_dir
            )
        
        # Run rocket models
        self.logger.info("Training time series models")
        
        # Find wrangled data
        wrangled_files = list(yochlol_dir.glob('wrangled_ts_*.parquet'))
        if not wrangled_files:
            raise FileNotFoundError("No wrangled time series files found")
            
        latest_wrangled = max(wrangled_files, key=lambda x: x.stat().st_mtime)
        
        self._run_in_environment(
            'yochlol',
            yochlol_dir / 'train.py',
            [
                '--data-path', str(latest_wrangled),
                '--rkt-chunksize', str(self.config['yochlol']['rocket_chunksize'])
            ],
            cwd=yochlol_dir
        )
        
        # Copy results to output directory
        results_dir = Path(self.config['pipeline']['output_dir']) / 'time_series_results'
        results_dir.mkdir(exist_ok=True)
        
        for result_file in yochlol_dir.glob('*.pkl'):
            shutil.copy2(result_file, results_dir)
        
        self.logger.info("Stage 2 completed successfully")
    
    def run_tabular_stage(self):
        """Stage 3: Run tabular modeling analysis"""
        self.logger.info("Starting Stage 3: Tabular Analysis")
        
        # Change to tabular_modeling directory and run scripts
        tabular_dir = Path(__file__).parent / 'tabular_modeling'
        
        # Download session data from S3
        self.logger.info("Downloading session data from S3")
        self._run_in_environment(
            'tabular',
            tabular_dir / 'download_s3_sessions.py',
            cwd=tabular_dir
        )
        
        # Process all sessions
        self.logger.info("Processing session data")
        self._run_in_environment(
            'tabular',
            tabular_dir / 'process_all_sessions.py',
            cwd=tabular_dir
        )
        
        # Fetch PCL scores
        self.logger.info("Fetching PCL scores")
        self._run_in_environment(
            'tabular',
            tabular_dir / 'fetch_pcl_scores.py',
            cwd=tabular_dir
        )
        
        # Run ML pipeline for each target
        for target in self.config['tabular_modeling']['targets']:
            self.logger.info(f"Running ML pipeline for target: {target}")
            
            args = [
                '--target', target,
                '--sample-fraction', str(self.config['tabular_modeling']['sample_fraction']),
                '--test-size', str(self.config['tabular_modeling']['test_size'])
            ]
            
            if self.config['tabular_modeling']['use_rfecv']:
                args.append('--rfecv')
                
            if self.config['tabular_modeling']['hyperparameter_tuning']:
                args.append('--tune')
            
            self._run_in_environment(
                'tabular',
                tabular_dir / 'ml_pipeline_with_pcl.py',
                args,
                cwd=tabular_dir
            )
        
        # Copy results to output directory
        results_dir = Path(self.config['pipeline']['output_dir']) / 'tabular_results'
        results_dir.mkdir(exist_ok=True)
        
        # Copy ML output
        ml_output_src = tabular_dir / 'ml_output'
        if ml_output_src.exists():
            ml_output_dst = results_dir / 'ml_output'
            if ml_output_dst.exists():
                shutil.rmtree(ml_output_dst)
            shutil.copytree(ml_output_src, ml_output_dst)
        
        self.logger.info("Stage 3 completed successfully")
    
    def wait_for_tabular_data(self):
        """Wait for external team to process data"""
        self.logger.info("Waiting for tabular data to be available...")
        input("Press Enter when tabular data is ready in AWS...")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        self.logger.info("Generating final report")
        
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
        self.logger.info("Starting full multi-environment pipeline execution")
        
        try:
            # Stage 1: Computer Vision (or skip if AWS mode)
            self.run_computer_vision_stage()
            
            # Stage 2: Time series analysis
            self.run_time_series_stage()
            
            # Wait for tabular data (unless we're in CV mode and uploaded data)
            if self.config['pipeline']['mode'] == 'aws':
                self.wait_for_tabular_data()
            
            # Stage 3: Tabular analysis
            self.run_tabular_stage()
            
            # Generate final report
            self.generate_final_report()
            
            self.logger.info("Pipeline execution completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Multi-Environment Automated MP Pipeline')
    parser.add_argument('--config', default='pipeline_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--stage', choices=['cv', 'time_series', 'tabular', 'full'],
                        default='full',
                        help='Which stage to run')
    parser.add_argument('--cohort',
                        help='Cohort name to use (overrides config file)')
    
    args = parser.parse_args()
    
    # Update config with cohort if specified
    if args.cohort:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['pipeline']['cohort'] = args.cohort
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Create pipeline instance
    pipeline = MultiEnvironmentPipeline(args.config)
    
    # Run requested stage
    if args.stage == 'cv':
        pipeline.run_computer_vision_stage()
    elif args.stage == 'time_series':
        pipeline.run_computer_vision_stage()
        pipeline.run_time_series_stage()
    elif args.stage == 'tabular':
        pipeline.run_tabular_stage()
    else:  # full
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()