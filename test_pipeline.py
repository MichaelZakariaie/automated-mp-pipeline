#!/usr/bin/env python3
"""
Comprehensive tests for the automated MP pipeline
Tests the integration between yochlol and tabular_modeling components
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

# Add the pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cohort_manager import CohortConfigManager
from cv_data_generator import ComputerVisionDataGenerator
from main_pipeline import AutomatedMPPipeline


class TestCohortConfigManager(unittest.TestCase):
    """Test cohort configuration management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test configurations
        self.main_config = {
            'pipeline': {
                'mode': 'aws',
                'cohort': 'test_cohort',
                'output_dir': './test_output'
            }
        }
        
        self.cohort_config = {
            'cohorts': {
                'test_cohort': {
                    'cohort_id': 99,
                    'description': 'Test cohort',
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
                        'available': ['task1', 'task2'],
                        'default': ['task1'],
                        'video_types': ['task1'],
                        'sample_counts': {'task1': 100, 'task2': 50}
                    },
                    'file_patterns': {
                        'session_file_pattern': '{session_id}_test.parquet',
                        'time_series_pattern': 'test_{cohort_id}_{timestamp}.parquet'
                    },
                    'ml_targets': {
                        'classification': 'test_ptsd_bin',
                        'regression': 'test_pcl_score'
                    }
                },
                'default': {
                    'cohort_id': 1,
                    'description': 'Default test cohort',
                    's3': {'compliance_bucket': 'default-bucket'},
                    'athena': {'database': 'default_db'},
                    'tasks': {'available': ['default_task']},
                    'file_patterns': {'session_file_pattern': 'default.parquet'},
                    'ml_targets': {'classification': 'ptsd_bin'}
                }
            }
        }
        
        # Write test configs
        self.main_config_path = self.temp_path / 'main_config.yaml'
        self.cohort_config_path = self.temp_path / 'cohort_config.yaml'
        
        with open(self.main_config_path, 'w') as f:
            yaml.dump(self.main_config, f)
        with open(self.cohort_config_path, 'w') as f:
            yaml.dump(self.cohort_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_cohort_config_loading(self):
        """Test that cohort configurations load correctly"""
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        self.assertEqual(manager.active_cohort, 'test_cohort')
        
        cohort_cfg = manager.get_cohort_config()
        self.assertEqual(cohort_cfg['cohort_id'], 99)
        self.assertEqual(cohort_cfg['description'], 'Test cohort')
    
    def test_s3_config_retrieval(self):
        """Test S3 configuration retrieval"""
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        s3_config = manager.get_s3_config()
        self.assertEqual(s3_config['compliance_bucket'], 'test-compliance')
        self.assertEqual(s3_config['data_bucket'], 'test-data')
        self.assertIn('{session_id}', s3_config['upload_prefix'])
    
    def test_athena_config_retrieval(self):
        """Test Athena configuration retrieval"""
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        athena_config = manager.get_athena_config()
        self.assertEqual(athena_config['database'], 'test_db')
        self.assertEqual(athena_config['tables']['pcl_scores'], 'test_pcl')
    
    def test_task_config_retrieval(self):
        """Test task configuration retrieval"""
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        task_config = manager.get_task_config()
        self.assertEqual(task_config['available'], ['task1', 'task2'])
        self.assertEqual(task_config['default'], ['task1'])
        self.assertEqual(task_config['sample_counts']['task1'], 100)
    
    def test_invalid_cohort_fallback(self):
        """Test fallback to default for invalid cohort"""
        # Update config to use invalid cohort
        self.main_config['pipeline']['cohort'] = 'nonexistent'
        with open(self.main_config_path, 'w') as f:
            yaml.dump(self.main_config, f)
        
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        # Should fall back to default
        cohort_cfg = manager.get_cohort_config()
        self.assertEqual(cohort_cfg['cohort_id'], 1)  # Default cohort ID
    
    def test_environment_variable_generation(self):
        """Test environment variable generation for subprocesses"""
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        yochlol_env = manager.get_yochlol_env_vars()
        self.assertEqual(yochlol_env['BUCKET_NAME'], 'test-compliance')
        self.assertEqual(yochlol_env['DATABASE'], 'test_db')
        self.assertEqual(yochlol_env['COHORT'], '99')
        
        tabular_env = manager.get_tabular_env_vars()
        self.assertEqual(tabular_env['S3_BUCKET'], 'test-data')
        self.assertEqual(tabular_env['PCL_TABLE'], 'test_pcl')
    
    def test_s3_path_formatting(self):
        """Test S3 path formatting with variables"""
        manager = CohortConfigManager(
            str(self.main_config_path),
            str(self.cohort_config_path)
        )
        
        template = "test/upload/{session_id}/data/"
        formatted = manager.format_s3_path(template, session_id='session123')
        self.assertEqual(formatted, "test/upload/session123/data/")


class TestCVDataGenerator(unittest.TestCase):
    """Test computer vision data generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_dummy_data_generation(self):
        """Test that dummy data is generated with correct structure"""
        generator = ComputerVisionDataGenerator(
            output_dir=str(self.temp_path),
            sessions_count=2,
            trials_per_session=10,
            fps=30
        )
        
        manifest = generator.generate_all_sessions()
        
        # Check manifest structure
        self.assertEqual(len(manifest['sessions']), 2)
        self.assertEqual(manifest['sessions_count'], 2)
        self.assertEqual(manifest['fps'], 30)
        
        # Check files were created
        manifest_file = self.temp_path / 'manifest.json'
        self.assertTrue(manifest_file.exists())
        
        # Check session data structure
        for session_info in manifest['sessions']:
            session_id = session_info['session_id']
            
            # Check trial data file exists and has correct structure
            trial_path = Path(session_info['trial_data_path'])
            self.assertTrue(trial_path.exists())
            
            trial_data = pd.read_parquet(trial_path)
            self.assertEqual(len(trial_data), 10)  # 10 trials
            self.assertIn('session_id', trial_data.columns)
            self.assertIn('trial', trial_data.columns)
            self.assertIn('dot_latency_pog', trial_data.columns)
            
            # Check time series data
            ts_path = Path(session_info['time_series_path'])
            self.assertTrue(ts_path.exists())
            
            ts_data = pd.read_parquet(ts_path)
            self.assertIn('timestamp', ts_data.columns)
            self.assertIn('gaze_x', ts_data.columns)
            self.assertIn('gaze_y', ts_data.columns)
    
    def test_data_schema_compatibility(self):
        """Test that generated data matches expected schema for downstream processing"""
        generator = ComputerVisionDataGenerator(
            output_dir=str(self.temp_path),
            sessions_count=1,
            trials_per_session=140  # Standard trial count
        )
        
        session_info = generator.generate_session()
        
        # Load and validate trial data
        trial_data = pd.read_parquet(session_info['trial_data_path'])
        
        # Check required columns for tabular modeling
        required_columns = [
            'dot_latency_pog', 'cue_latency_pog', 'trial_saccade_data_quality_pog',
            'cue_latency_pog_good', 'percent_bottom_freeface_pog', 
            'percent_top_freeface_pog', 'fixation_quality'
        ]
        
        for col in required_columns:
            self.assertIn(col, trial_data.columns, f"Missing required column: {col}")
        
        # Check data types and ranges
        self.assertTrue(trial_data['percent_bottom_freeface_pog'].between(0, 1).all())
        self.assertTrue(trial_data['percent_top_freeface_pog'].between(0, 1).all())
        self.assertIn(trial_data['trial_saccade_data_quality_pog'].iloc[0], ['good', 'bad'])
    
    def test_cohort_specific_file_naming(self):
        """Test that cohort-specific file patterns are used"""
        # Create a mock cohort manager
        temp_cohort_config = {
            'cohorts': {
                'test_cohort': {
                    'file_patterns': {
                        'session_file_pattern': 'test_{session_id}_custom.parquet'
                    }
                }
            }
        }
        
        cohort_config_path = self.temp_path / 'cohort_config.yaml'
        with open(cohort_config_path, 'w') as f:
            yaml.dump(temp_cohort_config, f)
        
        main_config_path = self.temp_path / 'main_config.yaml'
        with open(main_config_path, 'w') as f:
            yaml.dump({'pipeline': {'cohort': 'test_cohort'}}, f)
        
        cohort_manager = CohortConfigManager(
            str(main_config_path),
            str(cohort_config_path)
        )
        
        generator = ComputerVisionDataGenerator(
            output_dir=str(self.temp_path),
            sessions_count=1,
            cohort_manager=cohort_manager
        )
        
        session_info = generator.generate_session()
        trial_path = Path(session_info['trial_data_path'])
        
        # Check that filename follows cohort pattern
        self.assertTrue(trial_path.name.startswith('test_'))
        self.assertTrue(trial_path.name.endswith('_custom.parquet'))


class TestPipelineIntegration(unittest.TestCase):
    """Test main pipeline integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create minimal config
        self.config = {
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
            'execution': {
                'log_level': 'INFO',
                'log_file': str(self.temp_path / 'test.log')
            }
        }
        
        self.cohort_config = {
            'cohorts': {
                'test_cohort': {
                    'cohort_id': 99,
                    'description': 'Test cohort',
                    's3': {'compliance_bucket': 'test'},
                    'athena': {'database': 'test'},
                    'tasks': {'available': ['face_pairs'], 'default': ['face_pairs']},
                    'file_patterns': {'session_file_pattern': '{session_id}_test.parquet'},
                    'ml_targets': {'classification': 'ptsd_bin'}
                }
            }
        }
        
        self.config_path = self.temp_path / 'pipeline_config.yaml'
        self.cohort_config_path = self.temp_path / 'cohort_config.yaml'
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        with open(self.cohort_config_path, 'w') as f:
            yaml.dump(self.cohort_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly with cohort config"""
        # Change to temp directory to avoid interfering with real config
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            pipeline = AutomatedMPPipeline(str(self.config_path))
            
            self.assertEqual(pipeline.cohort_manager.active_cohort, 'test_cohort')
            self.assertTrue(Path(self.config['pipeline']['output_dir']).exists())
            self.assertTrue(Path(self.config['pipeline']['temp_dir']).exists())
            self.assertTrue(Path(self.config['pipeline']['reports_dir']).exists())
            
        finally:
            os.chdir(original_cwd)
    
    def test_cv_data_generation_integration(self):
        """Test that CV data generation works within pipeline context"""
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            pipeline = AutomatedMPPipeline(str(self.config_path))
            
            # Test the CV data generation part
            cv_config = pipeline.config['computer_vision']
            
            generator = ComputerVisionDataGenerator(
                output_dir=Path(pipeline.config['pipeline']['temp_dir']) / 'cv_output',
                sessions_count=cv_config['dummy_sessions_count'],
                fps=cv_config['fps'],
                cohort_manager=pipeline.cohort_manager
            )
            
            manifest = generator.generate_all_sessions()
            
            self.assertEqual(len(manifest['sessions']), 2)
            
            # Check that files exist in expected location
            cv_output_dir = Path(pipeline.config['pipeline']['temp_dir']) / 'cv_output'
            self.assertTrue(cv_output_dir.exists())
            
        finally:
            os.chdir(original_cwd)


class TestDataFormatCompatibility(unittest.TestCase):
    """Test compatibility between generated data and existing processing scripts"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_tabular_modeling_compatibility(self):
        """Test that generated data can be processed by tabular modeling scripts"""
        generator = ComputerVisionDataGenerator(
            output_dir=str(self.temp_path),
            sessions_count=3,
            trials_per_session=140
        )
        
        manifest = generator.generate_all_sessions()
        
        # Simulate the tabular modeling processing
        all_session_data = []
        
        for session_info in manifest['sessions']:
            trial_data = pd.read_parquet(session_info['trial_data_path'])
            
            # Test the transform_df_to_single_row logic (simplified version)
            trial_columns = [
                'dot_latency_pog', 'cue_latency_pog', 'trial_saccade_data_quality_pog',
                'cue_latency_pog_good', 'percent_bottom_freeface_pog', 
                'percent_top_freeface_pog', 'fixation_quality'
            ]
            
            new_data = {}
            
            # Check that we can process all 140 trials
            for trial_num in range(140):
                for col in trial_columns:
                    if col in trial_data.columns:
                        trial_row = trial_data[trial_data['trial'] == trial_num]
                        col_name = f'trial{trial_num}_{col}'
                        
                        if not trial_row.empty:
                            value = trial_row.iloc[0][col]
                            new_data[col_name] = value
                        else:
                            new_data[col_name] = None
            
            # Should have 140 trials * 7 features = 980 columns
            trial_feature_cols = [k for k in new_data.keys() if k.startswith('trial')]
            self.assertEqual(len(trial_feature_cols), 140 * len(trial_columns))
            
            all_session_data.append(new_data)
        
        # Should be able to create a combined dataframe
        combined_df = pd.DataFrame(all_session_data)
        self.assertEqual(len(combined_df), 3)  # 3 sessions
        
        # Check that we have the expected number of feature columns
        feature_cols = [col for col in combined_df.columns if col.startswith('trial')]
        self.assertEqual(len(feature_cols), 140 * len(trial_columns))
    
    def test_time_series_data_structure(self):
        """Test that time series data has the expected structure for yochlol processing"""
        generator = ComputerVisionDataGenerator(
            output_dir=str(self.temp_path),
            sessions_count=1,
            trials_per_session=10
        )
        
        session_info = generator.generate_session()
        ts_data = pd.read_parquet(session_info['time_series_path'])
        
        # Check required columns for time series analysis
        required_ts_columns = [
            'timestamp', 'gaze_x', 'gaze_y', 'pupil_diameter_left', 'pupil_diameter_right'
        ]
        
        for col in required_ts_columns:
            self.assertIn(col, ts_data.columns, f"Missing time series column: {col}")
        
        # Check data properties
        self.assertTrue(len(ts_data) > 0)
        self.assertTrue(ts_data['timestamp'].is_monotonic_increasing)
        self.assertTrue(ts_data['gaze_x'].between(-1, 1).all())
        self.assertTrue(ts_data['gaze_y'].between(-1, 1).all())


def run_integration_test():
    """Run a simple end-to-end integration test"""
    print("\n" + "="*60)
    print("RUNNING END-TO-END INTEGRATION TEST")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # Create test configuration
        config = {
            'pipeline': {
                'mode': 'local_cv',
                'cohort': 'integration_test',
                'output_dir': str(temp_path / 'output'),
                'temp_dir': str(temp_path / 'temp'),
                'reports_dir': str(temp_path / 'reports')
            },
            'computer_vision': {
                'use_dummy_data': True,
                'dummy_sessions_count': 3,
                'fps': 30
            },
            'execution': {
                'log_level': 'INFO',
                'log_file': str(temp_path / 'integration_test.log')
            }
        }
        
        cohort_config = {
            'cohorts': {
                'integration_test': {
                    'cohort_id': 999,
                    'description': 'Integration test cohort',
                    's3': {
                        'compliance_bucket': 'integration-test',
                        'data_bucket': 'integration-test',
                        'data_prefix': 'test/'
                    },
                    'athena': {
                        'database': 'integration_test',
                        'tables': {'pcl_scores': 'test_pcl'}
                    },
                    'tasks': {
                        'available': ['face_pairs'],
                        'default': ['face_pairs'],
                        'sample_counts': {'face_pairs': 140}
                    },
                    'file_patterns': {
                        'session_file_pattern': '{session_id}_integration_test.parquet'
                    },
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
        
        # Test the integration
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            print("1. Initializing pipeline...")
            pipeline = AutomatedMPPipeline(str(config_path))
            print("   ✓ Pipeline initialized successfully")
            
            print("2. Testing cohort configuration...")
            cohort_cfg = pipeline.cohort_manager.get_cohort_config()
            assert cohort_cfg['cohort_id'] == 999
            print("   ✓ Cohort configuration loaded correctly")
            
            print("3. Generating CV data...")
            cv_config = pipeline.config['computer_vision']
            generator = ComputerVisionDataGenerator(
                output_dir=Path(pipeline.config['pipeline']['temp_dir']) / 'cv_output',
                sessions_count=cv_config['dummy_sessions_count'],
                fps=cv_config['fps'],
                cohort_manager=pipeline.cohort_manager
            )
            
            manifest = generator.generate_all_sessions()
            assert len(manifest['sessions']) == 3
            print("   ✓ CV data generated successfully")
            
            print("4. Validating data compatibility...")
            session_info = manifest['sessions'][0]
            trial_data = pd.read_parquet(session_info['trial_data_path'])
            ts_data = pd.read_parquet(session_info['time_series_path'])
            
            assert len(trial_data) == 140  # Standard trial count
            assert 'session_id' in trial_data.columns
            assert len(ts_data) > 0
            assert 'timestamp' in ts_data.columns
            print("   ✓ Data format validation passed")
            
            print("5. Testing file naming patterns...")
            filename = Path(session_info['trial_data_path']).name
            assert 'integration_test' in filename
            print("   ✓ Cohort-specific file naming working")
            
            print("\n" + "="*60)
            print("INTEGRATION TEST PASSED ✓")
            print("="*60)
            print(f"Generated {len(manifest['sessions'])} sessions")
            print(f"Output directory: {temp_path}")
            print(f"Log file: {config['execution']['log_file']}")
            
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {str(e)}")
        raise
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    print("Running Automated MP Pipeline Tests")
    print("="*60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    run_integration_test()