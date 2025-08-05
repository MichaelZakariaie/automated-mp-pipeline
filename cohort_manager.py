#!/usr/bin/env python3
"""
Cohort Configuration Manager
Handles loading and applying cohort-specific configurations
"""

import yaml
import os
from pathlib import Path
import logging
from typing import Dict, Any, Optional


class CohortConfigManager:
    """Manages cohort-specific configurations"""
    
    def __init__(self, main_config_path='pipeline_config.yaml', 
                 cohort_config_path='cohort_config.yaml'):
        """Initialize with configuration paths"""
        self.main_config = self._load_yaml(main_config_path)
        self.cohort_config = self._load_yaml(cohort_config_path)
        self.logger = logging.getLogger(__name__)
        
        # Get active cohort from main config
        self.active_cohort = self.main_config['pipeline'].get('cohort', 'default')
        
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_cohort_config(self, cohort_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for specific cohort"""
        cohort = cohort_name or self.active_cohort
        
        if cohort not in self.cohort_config['cohorts']:
            self.logger.warning(f"Cohort '{cohort}' not found, using default")
            cohort = 'default'
            
        return self.cohort_config['cohorts'][cohort]
    
    def get_s3_config(self) -> Dict[str, str]:
        """Get S3 configuration for active cohort"""
        cohort_cfg = self.get_cohort_config()
        return cohort_cfg['s3']
    
    def get_athena_config(self) -> Dict[str, Any]:
        """Get Athena configuration for active cohort"""
        cohort_cfg = self.get_cohort_config()
        return cohort_cfg['athena']
    
    def get_task_config(self) -> Dict[str, Any]:
        """Get task configuration for active cohort"""
        cohort_cfg = self.get_cohort_config()
        return cohort_cfg['tasks']
    
    def get_file_patterns(self) -> Dict[str, str]:
        """Get file naming patterns for active cohort"""
        cohort_cfg = self.get_cohort_config()
        return cohort_cfg['file_patterns']
    
    def get_ml_targets(self) -> Dict[str, Any]:
        """Get ML target configuration for active cohort"""
        cohort_cfg = self.get_cohort_config()
        return cohort_cfg['ml_targets']
    
    def get_processing_params(self) -> Dict[str, Any]:
        """Get processing parameters for active cohort"""
        cohort = self.active_cohort
        processing = self.cohort_config.get('processing', {})
        
        if cohort in processing:
            return processing[cohort]
        else:
            return processing.get('default', {})
    
    def format_s3_path(self, template: str, **kwargs) -> str:
        """Format S3 path with cohort-specific values"""
        cohort_cfg = self.get_cohort_config()
        
        # Add cohort_id to kwargs if not present
        if 'cohort_id' not in kwargs:
            kwargs['cohort_id'] = cohort_cfg['cohort_id']
            
        return template.format(**kwargs)
    
    def get_yochlol_env_vars(self) -> Dict[str, str]:
        """Get environment variables for yochlol scripts"""
        s3_cfg = self.get_s3_config()
        athena_cfg = self.get_athena_config()
        task_cfg = self.get_task_config()
        cohort_cfg = self.get_cohort_config()
        
        return {
            'BUCKET_NAME': s3_cfg['compliance_bucket'],
            'S3_PREFIX': s3_cfg['compliance_prefix'],
            'DATABASE': athena_cfg['database'],
            'COHORT': str(cohort_cfg['cohort_id']),
            'TASK': ','.join(task_cfg['default']),
            'DEFAULT_VIDEO_TYPES': ','.join(task_cfg['video_types'])
        }
    
    def get_tabular_env_vars(self) -> Dict[str, str]:
        """Get environment variables for tabular modeling scripts"""
        s3_cfg = self.get_s3_config()
        athena_cfg = self.get_athena_config()
        
        return {
            'S3_BUCKET': s3_cfg['data_bucket'],
            'S3_PREFIX': s3_cfg['data_prefix'],
            'ATHENA_DATABASE': athena_cfg['database'],
            'PCL_TABLE': athena_cfg['tables']['pcl_scores']
        }
    
    def update_yochlol_config_files(self):
        """Update yochlol configuration files with cohort settings"""
        cohort_cfg = self.get_cohort_config()
        task_cfg = self.get_task_config()
        
        # Update insight_config.py
        insight_config_path = Path('yochlol/insight_config.py')
        if insight_config_path.exists():
            self.logger.info(f"Updating {insight_config_path} for cohort {self.active_cohort}")
            
            # Read current content
            with open(insight_config_path, 'r') as f:
                content = f.read()
            
            # Update COHORT value
            import re
            content = re.sub(
                r'COHORT\s*=\s*\d+',
                f'COHORT = {cohort_cfg["cohort_id"]}',
                content
            )
            
            # Update TASK value
            task_list = str(task_cfg['default'])
            content = re.sub(
                r'TASK\s*=\s*\[.*?\]',
                f'TASK = {task_list}',
                content,
                flags=re.DOTALL
            )
            
            # Write back
            with open(insight_config_path, 'w') as f:
                f.write(content)
    
    def print_cohort_summary(self):
        """Print summary of active cohort configuration"""
        cohort_cfg = self.get_cohort_config()
        s3_cfg = self.get_s3_config()
        athena_cfg = self.get_athena_config()
        task_cfg = self.get_task_config()
        
        print(f"\nActive Cohort Configuration: {self.active_cohort}")
        print("=" * 60)
        print(f"Cohort ID: {cohort_cfg['cohort_id']}")
        print(f"Description: {cohort_cfg['description']}")
        print(f"\nS3 Buckets:")
        print(f"  - Compliance: s3://{s3_cfg['compliance_bucket']}/{s3_cfg['compliance_prefix']}")
        print(f"  - Data: s3://{s3_cfg['data_bucket']}/{s3_cfg['data_prefix']}")
        print(f"\nAthena:")
        print(f"  - Database: {athena_cfg['database']}")
        print(f"  - PCL Table: {athena_cfg['tables']['pcl_scores']}")
        print(f"\nTasks:")
        print(f"  - Available: {', '.join(task_cfg['available'])}")
        print(f"  - Default: {', '.join(task_cfg['default'])}")
        print("=" * 60)


def main():
    """CLI for cohort configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cohort Configuration Manager')
    parser.add_argument('--cohort', help='Cohort name to use')
    parser.add_argument('--list', action='store_true', help='List available cohorts')
    parser.add_argument('--summary', action='store_true', help='Show cohort configuration summary')
    parser.add_argument('--update-configs', action='store_true', help='Update config files for cohort')
    
    args = parser.parse_args()
    
    manager = CohortConfigManager()
    
    if args.cohort:
        # Update main config with new cohort
        manager.active_cohort = args.cohort
        
    if args.list:
        print("\nAvailable Cohorts:")
        for cohort_name, cohort_cfg in manager.cohort_config['cohorts'].items():
            print(f"  - {cohort_name}: {cohort_cfg['description']}")
    
    if args.summary:
        manager.print_cohort_summary()
    
    if args.update_configs:
        manager.update_yochlol_config_files()
        print(f"\nConfiguration files updated for cohort: {manager.active_cohort}")


if __name__ == '__main__':
    main()