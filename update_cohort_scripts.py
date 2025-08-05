#!/usr/bin/env python3
"""
Script to update hardcoded values in yochlol and tabular_modeling scripts
with cohort-specific configurations
"""

import os
import re
import shutil
from pathlib import Path
import yaml
from cohort_manager import CohortConfigManager


class CohortScriptUpdater:
    """Updates scripts with cohort-specific values"""
    
    def __init__(self, cohort_name='default'):
        self.cohort_manager = CohortConfigManager()
        self.cohort_manager.active_cohort = cohort_name
        self.cohort_config = self.cohort_manager.get_cohort_config()
        self.s3_config = self.cohort_manager.get_s3_config()
        self.athena_config = self.cohort_manager.get_athena_config()
        self.task_config = self.cohort_manager.get_task_config()
        
    def backup_file(self, filepath):
        """Create backup of original file"""
        backup_path = f"{filepath}.backup"
        if not Path(backup_path).exists():
            shutil.copy2(filepath, backup_path)
            print(f"Created backup: {backup_path}")
            
    def update_yochlol_scripts(self):
        """Update yochlol directory scripts"""
        print("\nUpdating yochlol scripts...")
        
        # Update download_compliance_vids.py
        compliance_script = Path('yochlol/download_compliance_vids.py')
        if compliance_script.exists():
            self.backup_file(compliance_script)
            with open(compliance_script, 'r') as f:
                content = f.read()
            
            # Update bucket name
            content = re.sub(
                r'BUCKET_NAME\s*=\s*"[^"]*"',
                f'BUCKET_NAME = "{self.s3_config["compliance_bucket"]}"',
                content
            )
            
            # Update S3 prefix
            content = re.sub(
                r'S3_PREFIX\s*=\s*"[^"]*"',
                f'S3_PREFIX = "{self.s3_config["compliance_prefix"]}"',
                content
            )
            
            # Update video types
            video_types_str = str(self.task_config['video_types'])
            content = re.sub(
                r'DEFAULT_VIDEO_TYPES\s*=\s*\[.*?\]',
                f'DEFAULT_VIDEO_TYPES = {video_types_str}',
                content,
                flags=re.DOTALL
            )
            
            with open(compliance_script, 'w') as f:
                f.write(content)
            print(f"Updated: {compliance_script}")
        
        # Update get_data.py
        get_data_script = Path('yochlol/get_data.py')
        if get_data_script.exists():
            self.backup_file(get_data_script)
            with open(get_data_script, 'r') as f:
                content = f.read()
            
            # Update database references
            tables = self.athena_config['tables']
            
            # Update Athena table references
            content = re.sub(
                r'"data_quality\.qualtrics_surveys_unified"',
                f'"{self.athena_config["database"]}.{tables["surveys"]}"',
                content
            )
            
            content = re.sub(
                r'"data_quality\.mp_pcl_scores"',
                f'"{self.athena_config["database"]}.{tables["pcl_scores"]}"',
                content
            )
            
            content = re.sub(
                r'"data_quality\.mp_raps_scores"',
                f'"{self.athena_config["database"]}.{tables["raps_scores"]}"',
                content
            )
            
            content = re.sub(
                r'"data_quality\.master_query_session_completion_check"',
                f'"{self.athena_config["database"]}.{tables["session_completion"]}"',
                content
            )
            
            content = re.sub(
                r'"data_quality\.messy_prototyping_app_session_details"',
                f'"{self.athena_config["database"]}.{tables["app_session_details"]}"',
                content
            )
            
            # Update DATABASE constant
            content = re.sub(
                r'DATABASE\s*=\s*"[^"]*"',
                f'DATABASE = "{self.athena_config["database"]}"',
                content
            )
            
            with open(get_data_script, 'w') as f:
                f.write(content)
            print(f"Updated: {get_data_script}")
        
        # Update insight_config.py
        insight_config = Path('yochlol/insight_config.py')
        if insight_config.exists():
            self.backup_file(insight_config)
            with open(insight_config, 'r') as f:
                content = f.read()
            
            # Update COHORT
            content = re.sub(
                r'COHORT\s*=\s*\d+',
                f'COHORT = {self.cohort_config["cohort_id"]}',
                content
            )
            
            # Update TASK
            task_list = str(self.task_config['default'])
            content = re.sub(
                r'TASK\s*=\s*\[.*?\]',
                f'TASK = {task_list}',
                content,
                flags=re.DOTALL
            )
            
            with open(insight_config, 'w') as f:
                f.write(content)
            print(f"Updated: {insight_config}")
        
        # Update wrangle_ts.py
        wrangle_script = Path('yochlol/wrangle_ts.py')
        if wrangle_script.exists():
            self.backup_file(wrangle_script)
            with open(wrangle_script, 'r') as f:
                content = f.read()
            
            # Update TASK_SAMPLES dictionary
            task_samples = self.task_config['sample_counts']
            task_samples_str = "TASK_SAMPLES = {\n"
            for task, count in task_samples.items():
                task_samples_str += f'    "{task}": {count},\n'
            task_samples_str += "}"
            
            content = re.sub(
                r'TASK_SAMPLES\s*=\s*{[^}]*}',
                task_samples_str,
                content,
                flags=re.DOTALL
            )
            
            with open(wrangle_script, 'w') as f:
                f.write(content)
            print(f"Updated: {wrangle_script}")
    
    def update_tabular_modeling_scripts(self):
        """Update tabular_modeling directory scripts"""
        print("\nUpdating tabular_modeling scripts...")
        
        # Update download_s3_sessions.py
        download_script = Path('tabular_modeling/download_s3_sessions.py')
        if download_script.exists():
            self.backup_file(download_script)
            with open(download_script, 'r') as f:
                content = f.read()
            
            # Update bucket name
            content = re.sub(
                r"bucket_name\s*=\s*'[^']*'",
                f"bucket_name = '{self.s3_config['data_bucket']}'",
                content
            )
            
            # Update prefix
            content = re.sub(
                r"prefix\s*=\s*'[^']*'",
                f"prefix = '{self.s3_config['data_prefix']}'",
                content
            )
            
            # Update file pattern regex
            patterns = self.cohort_manager.get_file_patterns()
            session_pattern = patterns['session_file_pattern'].replace('{session_id}', '([^/]+)')
            session_pattern = session_pattern.replace('{timestamp}', '\\d+')
            
            new_regex = f"r'{self.s3_config['data_prefix']}[^/]+/face_pairs/processed_v2/{session_pattern}$'"
            content = re.sub(
                r"pattern\s*=\s*re\.compile\(r'[^']*'\)",
                f"pattern = re.compile({new_regex})",
                content
            )
            
            with open(download_script, 'w') as f:
                f.write(content)
            print(f"Updated: {download_script}")
        
        # Update fetch_pcl_scores.py
        fetch_script = Path('tabular_modeling/fetch_pcl_scores.py')
        if fetch_script.exists():
            self.backup_file(fetch_script)
            with open(fetch_script, 'r') as f:
                content = f.read()
            
            # Update database and table references
            content = re.sub(
                r"FROM\s+data_quality\.mp_pcl_scores",
                f"FROM {self.athena_config['database']}.{self.athena_config['tables']['pcl_scores']}",
                content
            )
            
            with open(fetch_script, 'w') as f:
                f.write(content)
            print(f"Updated: {fetch_script}")
        
        # Update test_aws_connection.py
        test_script = Path('tabular_modeling/test_aws_connection.py')
        if test_script.exists():
            self.backup_file(test_script)
            with open(test_script, 'r') as f:
                content = f.read()
            
            # Update database name
            content = re.sub(
                r"'data_quality'",
                f"'{self.athena_config['database']}'",
                content
            )
            
            # Update table name
            content = re.sub(
                r"'mp_pcl_scores'",
                f"'{self.athena_config['tables']['pcl_scores']}'",
                content
            )
            
            with open(test_script, 'w') as f:
                f.write(content)
            print(f"Updated: {test_script}")
    
    def restore_backups(self):
        """Restore original files from backups"""
        print("\nRestoring from backups...")
        
        for directory in ['yochlol', 'tabular_modeling']:
            for backup_file in Path(directory).glob('**/*.backup'):
                original_file = str(backup_file).replace('.backup', '')
                shutil.copy2(backup_file, original_file)
                print(f"Restored: {original_file}")
    
    def print_summary(self):
        """Print summary of changes"""
        print("\n" + "=" * 60)
        print(f"Cohort Configuration Applied: {self.cohort_manager.active_cohort}")
        print("=" * 60)
        print("\nKey values updated:")
        print(f"  Cohort ID: {self.cohort_config['cohort_id']}")
        print(f"  S3 Compliance: s3://{self.s3_config['compliance_bucket']}/{self.s3_config['compliance_prefix']}")
        print(f"  S3 Data: s3://{self.s3_config['data_bucket']}/{self.s3_config['data_prefix']}")
        print(f"  Athena DB: {self.athena_config['database']}")
        print(f"  PCL Table: {self.athena_config['tables']['pcl_scores']}")
        print(f"  Default Tasks: {', '.join(self.task_config['default'])}")
        print("\nBackup files created with .backup extension")
        print("Run with --restore to revert to original files")


def main():
    """CLI for updating scripts with cohort configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update scripts with cohort-specific configuration'
    )
    parser.add_argument('--cohort', default='default',
                        help='Cohort name to apply')
    parser.add_argument('--restore', action='store_true',
                        help='Restore original files from backups')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be changed without modifying files')
    
    args = parser.parse_args()
    
    updater = CohortScriptUpdater(args.cohort)
    
    if args.restore:
        updater.restore_backups()
    elif args.dry_run:
        print("DRY RUN - No files will be modified")
        updater.print_summary()
    else:
        # Update all scripts
        updater.update_yochlol_scripts()
        updater.update_tabular_modeling_scripts()
        updater.print_summary()


if __name__ == '__main__':
    main()