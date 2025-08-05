#!/usr/bin/env python3
"""
Run all 4 variations of ML pipeline and generate combined report
1. Regular (single train/test split)
2. Regular with PCL filtering
3. K-fold cross-validation
4. K-fold with PCL filtering
"""

import subprocess
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
import time
import yaml
import os

class CombinedPipelineRunner:
    """Run all ML pipeline variations and generate combined report"""
    
    def __init__(self, target='ptsd_bin', n_folds=5, n_iterations=5, sample_fraction=1.0, 
                 pca_components=None, config=None):
        # If config provided, override with config values
        if config:
            self.target = config.get('target', target)
            self.n_folds = config.get('n_folds', n_folds)
            self.n_iterations = config.get('n_iterations', n_iterations)
            self.sample_fraction = config.get('sample_fraction', sample_fraction)
            self.pca_components = config.get('pca_components', pca_components or [])
            self.variations = config.get('variations', {
                'regular': True,
                'regular_filtered': True,
                'kfold': True,
                'kfold_filtered': True
            })
            self.pcl_filtering = config.get('pcl_filtering', {
                'remove_intermediate': True,
                'ptsd_threshold': 33,
                'buffer': 8
            })
            self.output_config = config.get('output', {
                'base_dir': 'ml_output',
                'generate_report': True
            })
            self.scp_config = config.get('scp', {
                'enabled': False,
                'destination': '/Users/michaelzakariaie/Desktop'
            })
            self.advanced = config.get('advanced', {
                'random_state': 42,
                'test_size': 0.2,
                'delay_between_runs': 2,
                'n_jobs': -1,
                'verbose': 1
            })
            # Feature engineering settings
            self.feature_engineering = config.get('feature_engineering', {
                'use_rfecv': False,
                'rfecv_step': 1,
                'rfecv_min_features': 5
            })
            # Model selection settings
            self.models_config = config.get('models', {
                'include_models': None,
                'exclude_models': []
            })
            # Hyperparameter tuning settings
            self.hyperparameter_tuning = config.get('hyperparameter_tuning', {
                'enabled': False,
                'use_random_search': True,
                'n_iter': 30,
                'top_models': 3,
                'models_to_tune': ['XGBoost', 'Random Forest', 'Gradient Boosting']
            })
            # Automated pipeline settings
            self.automated_pipeline = config.get('automated_pipeline', {
                'enabled': False
            })
            # Check for multi-target configuration
            self.multi_target_config = config.get('multi_target', None)
        else:
            self.target = target
            self.n_folds = n_folds
            self.n_iterations = n_iterations
            self.sample_fraction = sample_fraction
            self.pca_components = pca_components if pca_components else []
            self.variations = {
                'regular': True,
                'regular_filtered': True,
                'kfold': True,
                'kfold_filtered': True
            }
            self.pcl_filtering = {
                'remove_intermediate': True,
                'ptsd_threshold': 33,
                'buffer': 8
            }
            self.output_config = {
                'base_dir': 'ml_output',
                'generate_report': True
            }
            self.scp_config = {
                'enabled': False,
                'destination': '/Users/michaelzakariaie/Desktop'
            }
            self.advanced = {
                'random_state': 42,
                'test_size': 0.2,
                'delay_between_runs': 2,
                'n_jobs': -1,
                'verbose': 1
            }
            self.feature_engineering = {
                'use_rfecv': False,
                'rfecv_step': 1,
                'rfecv_min_features': 5
            }
            self.models_config = {
                'include_models': None,
                'exclude_models': []
            }
            self.hyperparameter_tuning = {
                'enabled': False,
                'use_random_search': True,
                'n_iter': 30,
                'top_models': 3,
                'models_to_tune': ['XGBoost', 'Random Forest', 'Gradient Boosting']
            }
            self.automated_pipeline = {
                'enabled': False
            }
            self.multi_target_config = None
        
        self.results = {}
        self.run_folders = {}
        
    def run_pipeline(self, pipeline_type, remove_intermediate_pcl=False, pca_components=None):
        """Run a specific pipeline variation"""
        print(f"\n{'='*60}")
        print(f"Running {pipeline_type} pipeline")
        if remove_intermediate_pcl:
            lower = self.pcl_filtering['ptsd_threshold'] - self.pcl_filtering['buffer']
            upper = self.pcl_filtering['ptsd_threshold'] + self.pcl_filtering['buffer']
            print(f"With PCL filtering (removing scores {lower}-{upper})")
        if pca_components:
            print(f"With PCA: {pca_components} components")
        print(f"{'='*60}\n")
        
        # Build command
        if pipeline_type == 'regular':
            cmd = ['python', 'ml_pipeline_with_pcl.py', '--target', self.target,
                   '--test-size', str(self.advanced['test_size'])]
        else:  # k-fold
            cmd = ['python', 'ml_pipeline_kfold.py', '--target', self.target,
                   '--n-folds', str(self.n_folds), '--n-iterations', str(self.n_iterations),
                   '--random-state', str(self.advanced['random_state'])]
        
        # Add automated pipeline flag if enabled
        if self.automated_pipeline['enabled']:
            cmd.append('--auto')
        else:
            # Add individual feature flags
            if self.feature_engineering['use_rfecv']:
                cmd.append('--rfecv')
            if self.hyperparameter_tuning['enabled']:
                cmd.append('--tune')
        
        if remove_intermediate_pcl:
            cmd.extend(['--remove-intermediate-pcl',
                       '--ptsd-threshold', str(self.pcl_filtering['ptsd_threshold']),
                       '--buffer', str(self.pcl_filtering['buffer'])])
        
        # Add sample fraction if less than 1
        if self.sample_fraction < 1.0:
            cmd.extend(['--sample-fraction', str(self.sample_fraction)])
        
        # Add PCA if specified
        if pca_components:
            cmd.extend(['--use-pca', '--n-components', str(pca_components)])
        
        # Add model configuration
        if self.models_config['include_models']:
            cmd.extend(['--include-models'] + self.models_config['include_models'])
        if self.models_config['exclude_models']:
            cmd.extend(['--exclude-models'] + self.models_config['exclude_models'])
        
        # Run pipeline
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print subprocess output for debugging
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])  # Last 500 chars
            
            # Extract run folder from output
            output = result.stdout + '\n' + result.stderr
            found_folder = None
            
            for line in output.split('\n'):
                if 'Saving results to:' in line:
                    # Extract path after "Saving results to: "
                    path_part = line.split('Saving results to:')[-1].strip()
                    if 'ml_output' in path_part:
                        # Use the full path as returned
                        found_folder = path_part
                        break
            
            if found_folder:
                return found_folder
            
            # If not found, try to find most recent run folder
            from pathlib import Path
            ml_output = Path('ml_output')
            if ml_output.exists():
                # Get most recent folder (within last 10 seconds to avoid picking up old runs)
                import time
                current_time = time.time()
                recent_folders = []
                
                for folder in ml_output.iterdir():
                    if folder.is_dir() and folder.name.startswith('run_'):
                        # Check if folder was created within last 30 seconds
                        if (current_time - folder.stat().st_mtime) < 30:
                            recent_folders.append(folder)
                
                if recent_folders:
                    # Sort by modification time, get most recent
                    most_recent = sorted(recent_folders, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    return str(most_recent)
            
            print(f"Could not find output folder. Last 10 lines of output:")
            print('\n'.join(output.split('\n')[-10:]))
                
        except Exception as e:
            print(f"Error running {pipeline_type}: {e}")
            
        return None
    
    def run_all_variations(self):
        """Run all pipeline variations including PCA"""
        # Check if we have multiple targets configuration
        if hasattr(self, 'multi_target_config') and self.multi_target_config:
            self._run_multi_target_variations()
        else:
            # Original single target logic
            self._run_single_target_variations()
    
    def _run_single_target_variations(self):
        """Original method for single target runs"""
        # Build variations based on config
        base_variations = []
        if self.variations.get('regular', True):
            base_variations.append(('regular', False, None))
        if self.variations.get('regular_filtered', True):
            base_variations.append(('regular_filtered', True, None))
        if self.variations.get('kfold', True):
            base_variations.append(('kfold', False, None))
        if self.variations.get('kfold_filtered', True):
            base_variations.append(('kfold_filtered', True, None))
        
        # Add PCA variations if specified
        all_variations = base_variations.copy()
        for n_comp in self.pca_components:
            all_variations.extend([
                (f'regular_pca{n_comp}', False, n_comp),
                (f'regular_filtered_pca{n_comp}', True, n_comp),
                (f'kfold_pca{n_comp}', False, n_comp),
                (f'kfold_filtered_pca{n_comp}', True, n_comp)
            ])
        
        print(f"Running {len(all_variations)} pipeline variations...")
        
        for name, use_filter, pca_comp in all_variations:
            pipeline_type = 'regular' if 'regular' in name else 'kfold'
            run_folder = self.run_pipeline(pipeline_type, use_filter, pca_comp)
            
            if run_folder:
                self.run_folders[name] = run_folder
                print(f"✓ Completed {name}: {run_folder}")
            else:
                print(f"✗ Failed {name}")
            
            # Small delay between runs
            time.sleep(self.advanced['delay_between_runs'])
    
    def _run_multi_target_variations(self):
        """Run variations for multiple targets as configured"""
        print(f"Running multi-target analysis...")
        
        for target_config in self.multi_target_config:
            target = target_config['target']
            variations = target_config.get('variations', {})
            
            print(f"\n{'='*60}")
            print(f"TARGET: {target}")
            print(f"{'='*60}")
            
            # Temporarily set the target
            original_target = self.target
            self.target = target
            
            # Build variations for this target
            all_variations = []
            
            if variations.get('regular', False):
                all_variations.append((f'{target}_regular', False, None))
            if variations.get('regular_filtered', False):
                all_variations.append((f'{target}_regular_filtered', True, None))
            if variations.get('kfold', False):
                all_variations.append((f'{target}_kfold', False, None))
            if variations.get('kfold_filtered', False):
                all_variations.append((f'{target}_kfold_filtered', True, None))
            
            # Run variations for this target
            for name, use_filter, pca_comp in all_variations:
                pipeline_type = 'regular' if 'regular' in name else 'kfold'
                run_folder = self.run_pipeline(pipeline_type, use_filter, pca_comp)
                
                if run_folder:
                    self.run_folders[name] = run_folder
                    print(f"✓ Completed {name}: {run_folder}")
                else:
                    print(f"✗ Failed {name}")
                
                # Small delay between runs
                time.sleep(self.advanced['delay_between_runs'])
            
            # Restore original target
            self.target = original_target
    
    def generate_combined_report(self, show_scp=None, scp_dest=None):
        """Generate combined HTML report with tabs for all variations"""
        if not self.run_folders:
            print("No successful runs to report")
            return None
        
        # Use config values if not overridden
        if show_scp is None:
            show_scp = self.scp_config['enabled']
        if scp_dest is None:
            scp_dest = self.scp_config['destination']
        
        # Create new output folder for combined report
        from run_counter import get_next_run_folder
        output_path, run_number = get_next_run_folder(self.output_config['base_dir'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Import report generator
        from report_generator import ReportGenerator
        from combined_report_generator import CombinedReportGenerator
        
        generator = CombinedReportGenerator(output_path)
        generator.generate_combined_report(self.run_folders)
        
        print(f"\n{'='*60}")
        print(f"Combined report saved to: {output_path}")
        print(f"{'='*60}")
        
        if show_scp and scp_dest:
            source_path = output_path.absolute()
            scp_command = f"scp -r michael@192.168.0.119:{source_path} {scp_dest}"
            
            print(f"\nTo transfer the report to your local machine, run this command from your local terminal:")
            print(f"\n  {scp_command}\n")
            print(f"Report folder: {output_path.name}/")
            print(f"{'='*60}\n")
        
        return output_path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run all ML pipeline variations',
        epilog='You can either use command line arguments or provide a YAML config file.'
    )
    
    # Add config file option
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    
    # Original command line arguments (these override config file if both provided)
    parser.add_argument('--target', choices=['ptsd_bin', 'pcl_score'],
                        help='Target variable for prediction')
    parser.add_argument('--n-folds', type=int,
                        help='Number of folds for k-fold cross-validation')
    parser.add_argument('--n-iterations', type=int,
                        help='Number of k-fold iterations')
    parser.add_argument('--sample-fraction', type=float,
                        help='Fraction of data to use (for faster testing)')
    parser.add_argument('--pca-components', type=int, nargs='*',
                        help='List of PCA component numbers to test')
    parser.add_argument('--scp', action='store_true',
                        help='Show SCP command to transfer report after completion')
    parser.add_argument('--scp-dest',
                        help='Destination path for SCP')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    
    # Command line args override config file
    if config:
        # Override config with any command line arguments provided
        if args.target:
            config['target'] = args.target
        if args.n_folds is not None:
            config['n_folds'] = args.n_folds
        if args.n_iterations is not None:
            config['n_iterations'] = args.n_iterations
        if args.sample_fraction is not None:
            config['sample_fraction'] = args.sample_fraction
        if args.pca_components is not None:
            config['pca_components'] = args.pca_components
        if args.scp:
            config['scp']['enabled'] = True
        if args.scp_dest:
            config['scp']['destination'] = args.scp_dest
        
        runner = CombinedPipelineRunner(config=config)
    else:
        # Use command line arguments with defaults
        runner = CombinedPipelineRunner(
            target=args.target or 'ptsd_bin',
            n_folds=args.n_folds or 5,
            n_iterations=args.n_iterations or 5,
            sample_fraction=args.sample_fraction or 1.0,
            pca_components=args.pca_components or []
        )
    
    # Run all variations
    runner.run_all_variations()
    
    # Generate combined report if configured
    if not config or config.get('output', {}).get('generate_report', True):
        runner.generate_combined_report(
            show_scp=args.scp if args.scp else None,
            scp_dest=args.scp_dest if args.scp_dest else None
        )

if __name__ == "__main__":
    main()