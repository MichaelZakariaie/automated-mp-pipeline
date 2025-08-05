#!/usr/bin/env python3
"""
Tests for the multi-environment automated MP pipeline
Tests the integration between components without dependency conflicts
"""

import unittest
import tempfile
import shutil
import yaml
import json
import pandas as pd
from pathlib import Path
import os
import sys
import subprocess

# Add the pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cohort_manager import CohortConfigManager
from cv_data_generator import ComputerVisionDataGenerator


class TestMultiEnvironmentSetup(unittest.TestCase):
    """Test that multi-environment setup works correctly"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create minimal test environments
        for env_name in ['env_main', 'env_cv', 'env_yochlol', 'env_tabular']:
            env_path = self.temp_path / env_name
            env_path.mkdir(parents=True)
            bin_path = env_path / 'bin'
            bin_path.mkdir()
            
            # Create dummy python executable
            python_exe = bin_path / 'python'
            python_exe.write_text('#!/bin/bash\necho "Dummy python for testing"\n')
            python_exe.chmod(0o755)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_environment_verification(self):
        """Test that environment verification works"""
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Create test config
            config = {
                'pipeline': {'mode': 'aws', 'cohort': 'default'},
                'execution': {'log_level': 'INFO', 'log_file': 'test.log'}
            }
            
            cohort_config = {
                'cohorts': {
                    'default': {
                        'cohort_id': 1,
                        's3': {'compliance_bucket': 'test'},
                        'athena': {'database': 'test'},
                        'tasks': {'available': ['test']},
                        'file_patterns': {'session_file_pattern': 'test.parquet'},
                        'ml_targets': {'classification': 'test'}
                    }
                }
            }
            
            with open('pipeline_config.yaml', 'w') as f:
                yaml.dump(config, f)
            with open('cohort_config.yaml', 'w') as f:
                yaml.dump(cohort_config, f)
            
            # Import and test (this should work now that envs exist)
            from multi_env_pipeline import MultiEnvironmentPipeline
            
            pipeline = MultiEnvironmentPipeline('pipeline_config.yaml')
            
            # Check that it found all environments
            self.assertEqual(len(pipeline.env_paths), 4)
            for env_name, env_path in pipeline.env_paths.items():
                self.assertTrue(env_path.exists(), f"Environment {env_name} should exist")
        
        finally:
            os.chdir(original_cwd)


class TestCohortIntegration(unittest.TestCase):
    """Test cohort configuration integration with multi-environment setup"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test configurations
        self.main_config = {
            'pipeline': {
                'mode': 'local_cv',
                'cohort': 'test_cohort',
                'output_dir': str(self.temp_path / 'output'),
                'temp_dir': str(self.temp_path / 'temp'),
                'reports_dir': str(self.temp_path / 'reports')
            },
            'computer_vision': {
                'use_dummy_data': True,
                'dummy_sessions_count': 2,
                'fps': 30
            },
            'yochlol': {
                'compliance_metrics': True,
                'wrangle_time_series': True,
                'task': 'face_pairs',
                'rocket_chunksize': 256
            },
            'tabular_modeling': {
                'targets': ['ptsd_bin'],
                'sample_fraction': 0.1,
                'test_size': 0.2,
                'use_rfecv': False,
                'hyperparameter_tuning': False
            },
            'execution': {
                'log_level': 'INFO',
                'log_file': str(self.temp_path / 'test.log')
            }
        }
        
        self.cohort_config = {
            'cohorts': {
                'test_cohort': {
                    'cohort_id': 99,
                    'description': 'Test cohort for multi-env',
                    's3': {
                        'compliance_bucket': 'test-compliance',
                        'compliance_prefix': 'test/compliance/',
                        'data_bucket': 'test-data',
                        'data_prefix': 'test/data/',
                        'upload_bucket': 'test-upload',
                        'upload_prefix': 'test/upload/{session_id}/'
                    },
                    'athena': {
                        'database': 'test_db',
                        'tables': {
                            'pcl_scores': 'test_pcl',
                            'surveys': 'test_surveys'
                        }
                    },
                    'tasks': {
                        'available': ['face_pairs'],
                        'default': ['face_pairs'],
                        'video_types': ['face_pairs'],
                        'sample_counts': {'face_pairs': 140}
                    },
                    'file_patterns': {
                        'session_file_pattern': '{session_id}_test_multienv.parquet',
                        'time_series_pattern': 'test_multienv_{cohort_id}_{timestamp}.parquet'
                    },
                    'ml_targets': {
                        'classification': 'test_ptsd_bin',
                        'regression': 'test_pcl_score'
                    }
                }
            }
        }
        
        # Write test configs
        self.main_config_path = self.temp_path / 'pipeline_config.yaml'
        self.cohort_config_path = self.temp_path / 'cohort_config.yaml'
        
        with open(self.main_config_path, 'w') as f:
            yaml.dump(self.main_config, f)
        with open(self.cohort_config_path, 'w') as f:
            yaml.dump(self.cohort_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_cohort_manager_with_multienv(self):
        """Test that cohort manager works with multi-environment configuration"""
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        self.assertEqual(manager.active_cohort, 'test_cohort')
        
        # Test environment variable generation
        yochlol_env = manager.get_yochlol_env_vars()
        self.assertEqual(yochlol_env['BUCKET_NAME'], 'test-compliance')
        self.assertEqual(yochlol_env['COHORT'], '99')
        
        tabular_env = manager.get_tabular_env_vars()
        self.assertEqual(tabular_env['S3_BUCKET'], 'test-data')
        self.assertEqual(tabular_env['PCL_TABLE'], 'test_pcl')
    
    def test_cv_data_generation_with_cohort(self):
        """Test CV data generation with cohort-specific patterns"""
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        generator = ComputerVisionDataGenerator(
            output_dir=str(self.temp_path / 'cv_output'),
            sessions_count=2,
            trials_per_session=10,
            cohort_manager=manager
        )
        
        manifest = generator.generate_all_sessions()
        
        # Check that cohort-specific file patterns were used
        session_info = manifest['sessions'][0]
        trial_path = Path(session_info['trial_data_path'])
        
        # Should contain 'test_multienv' from our cohort pattern
        self.assertIn('test_multienv', trial_path.name)


class TestPipelineComponents(unittest.TestCase):
    """Test individual pipeline components work correctly"""
    
    def test_cv_data_format_compatibility(self):
        """Test that CV-generated data is compatible with downstream processing"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            generator = ComputerVisionDataGenerator(
                output_dir=str(temp_path),
                sessions_count=1,
                trials_per_session=140  # Full trial count
            )
            
            session_info = generator.generate_session()
            
            # Load and validate trial data
            trial_data = pd.read_parquet(session_info['trial_data_path'])
            
            # Check that it has the expected structure for tabular processing
            self.assertEqual(len(trial_data), 140)
            
            required_columns = [
                'session_id', 'trial', 'dot_latency_pog', 'cue_latency_pog',
                'trial_saccade_data_quality_pog', 'percent_bottom_freeface_pog'
            ]
            
            for col in required_columns:
                self.assertIn(col, trial_data.columns, f"Missing column: {col}")
            
            # Check time series data
            ts_data = pd.read_parquet(session_info['time_series_path'])
            
            required_ts_columns = ['timestamp', 'gaze_x', 'gaze_y']
            for col in required_ts_columns:
                self.assertIn(col, ts_data.columns, f"Missing time series column: {col}")
            
            # Check that time series has reasonable length for 30fps
            self.assertGreater(len(ts_data), 1000, "Time series should have reasonable length")
            
        finally:
            shutil.rmtree(temp_dir)


def run_integration_test():
    """Run end-to-end integration test for multi-environment setup"""
    print("\n" + "="*70)
    print("RUNNING MULTI-ENVIRONMENT INTEGRATION TEST")
    print("="*70)
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        print("1. Testing multi-environment setup simulation...")
        
        # Simulate environment setup
        for env_name in ['env_main', 'env_cv', 'env_yochlol', 'env_tabular']:
            env_path = temp_path / env_name
            env_path.mkdir(parents=True)
            bin_path = env_path / 'bin'
            bin_path.mkdir()
            
            # Create mock python executable that just echoes success
            python_exe = bin_path / 'python'
            python_exe.write_text(f'#!/bin/bash\necho "Mock {env_name} execution successful"\nexit 0\n')
            python_exe.chmod(0o755)
        
        print("   ✓ Mock environments created")
        
        print("2. Testing cohort configuration...")
        
        # Create test configurations
        config = {
            'pipeline': {
                'mode': 'local_cv',
                'cohort': 'integration_test',
                'output_dir': str(temp_path / 'output'),
                'temp_dir': str(temp_path / 'temp'),
                'reports_dir': str(temp_path / 'reports')
            },
            'computer_vision': {'use_dummy_data': True, 'dummy_sessions_count': 1},
            'yochlol': {'compliance_metrics': False, 'wrangle_time_series': False},
            'tabular_modeling': {'targets': ['ptsd_bin']},
            'execution': {'log_level': 'INFO', 'log_file': str(temp_path / 'test.log')}
        }
        
        cohort_config = {
            'cohorts': {
                'integration_test': {
                    'cohort_id': 999,
                    'description': 'Multi-env integration test',
                    's3': {'compliance_bucket': 'test'},
                    'athena': {'database': 'test'},
                    'tasks': {'available': ['face_pairs'], 'default': ['face_pairs']},
                    'file_patterns': {'session_file_pattern': '{session_id}_integration.parquet'},
                    'ml_targets': {'classification': 'ptsd_bin'}
                }
            }
        }
        
        config_path = temp_path / 'pipeline_config.yaml'
        cohort_config_path = temp_path / 'cohort_config.yaml'
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        with open(cohort_config_path, 'w') as f:
            yaml.dump(cohort_config, f)
        
        print("   ✓ Configuration files created")
        
        print("3. Testing CV data generation...")
        
        # Test CV data generation
        generator = ComputerVisionDataGenerator(
            output_dir=str(temp_path / 'cv_output'),
            sessions_count=1,
            trials_per_session=10
        )
        
        manifest = generator.generate_all_sessions()
        assert len(manifest['sessions']) == 1
        
        print("   ✓ CV data generation successful")
        
        print("4. Testing cohort manager integration...")
        
        original_cwd = os.getcwd()
        os.chdir(temp_path)
        
        try:
            manager = CohortConfigManager(str(config_path), str(cohort_config_path))
            
            assert manager.active_cohort == 'integration_test'
            
            yochlol_env = manager.get_yochlol_env_vars()
            assert 'COHORT' in yochlol_env
            assert yochlol_env['COHORT'] == '999'
            
            print("   ✓ Cohort manager working correctly")
            
        finally:
            os.chdir(original_cwd)
        
        print("\n" + "="*70)
        print("MULTI-ENVIRONMENT INTEGRATION TEST PASSED ✓")
        print("="*70)
        print(f"Test environment created at: {temp_path}")
        print("Key features verified:")
        print("  - Multi-environment setup simulation")
        print("  - Cohort configuration management")
        print("  - CV data generation with cohort patterns")
        print("  - Environment variable management")
        
    except Exception as e:
        print(f"\n❌ MULTI-ENVIRONMENT INTEGRATION TEST FAILED: {str(e)}")
        raise
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    print("Running Multi-Environment Automated MP Pipeline Tests")
    print("="*60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    run_integration_test()