#!/usr/bin/env python3
"""
Download face_pairs session files from S3 bucket
"""

import boto3
import os
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def list_s3_sessions(bucket_name='senseye-data-quality'):
    """List all session files matching the pattern"""
    s3 = boto3.client('s3')
    sessions = []
    
    print("Listing files in S3 bucket...")
    
    # List all objects with the prefix
    prefix = 'messy_prototyping_saturn_uploads/'
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    pattern = re.compile(r'messy_prototyping_saturn_uploads/([^/]+)/face_pairs/processed_v2/[^/]+_inter_pir_face_pairs_v2_latedwell_pog\.parquet$')
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                match = pattern.match(key)
                if match:
                    session_id = match.group(1)
                    sessions.append({
                        'session_id': session_id,
                        's3_key': key,
                        'size': obj['Size']
                    })
    
    return sessions

def download_file(s3_client, bucket_name, s3_key, local_path):
    """Download a single file from S3"""
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True, s3_key
    except Exception as e:
        return False, f"{s3_key}: {str(e)}"

def download_sessions(sessions, output_dir='downloaded_sessions', max_workers=10):
    """Download all session files from S3 using multiple threads"""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    s3 = boto3.client('s3')
    bucket_name = 'senseye-data-quality'
    
    # Prepare download tasks
    download_tasks = []
    for session in sessions:
        # Create filename from session_id
        filename = f"{session['session_id']}_inter_pir_face_pairs_v2_latedwell_pog.parquet"
        local_path = os.path.join(output_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(local_path) and os.path.getsize(local_path) == session['size']:
            print(f"Skipping {filename} (already downloaded)")
            continue
            
        download_tasks.append({
            's3_key': session['s3_key'],
            'local_path': local_path,
            'session_id': session['session_id']
        })
    
    if not download_tasks:
        print("All files already downloaded!")
        return
    
    print(f"\nDownloading {len(download_tasks)} files...")
    
    # Download files in parallel
    successful = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_task = {
            executor.submit(download_file, s3, bucket_name, task['s3_key'], task['local_path']): task
            for task in download_tasks
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=len(download_tasks), desc="Downloading") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                success, result = future.result()
                
                if success:
                    successful += 1
                else:
                    failed.append(result)
                
                pbar.update(1)
                pbar.set_postfix({'success': successful, 'failed': len(failed)})
    
    # Report results
    print(f"\n✓ Successfully downloaded: {successful} files")
    if failed:
        print(f"✗ Failed downloads: {len(failed)}")
        for error in failed[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

def verify_aws_access():
    """Verify AWS credentials and S3 access"""
    try:
        # Test AWS credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        print(f"AWS User/Role: {identity['Arn']}")
        
        # Test S3 access
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket='senseye-data-quality')
        print("✓ S3 access verified")
        return True
    except Exception as e:
        print(f"Error: Unable to access AWS: {e}")
        print("\nPlease configure AWS credentials:")
        print("  1. Run: aws configure")
        print("  2. Or set environment variables:")
        print("     export AWS_ACCESS_KEY_ID=your_key")
        print("     export AWS_SECRET_ACCESS_KEY=your_secret")
        print("     export AWS_DEFAULT_REGION=us-east-1")
        return False

def main():
    """Main function"""
    # Check AWS credentials
    if not verify_aws_access():
        return
    
    # List sessions
    sessions = list_s3_sessions()
    
    if not sessions:
        print("No sessions found matching the pattern!")
        return
    
    print(f"\nFound {len(sessions)} sessions to download")
    print(f"Total size: {sum(s['size'] for s in sessions) / 1024 / 1024:.1f} MB")
    
    # Show sample sessions
    print("\nSample sessions:")
    for session in sessions[:5]:
        print(f"  - {session['session_id']} ({session['size'] / 1024:.1f} KB)")
    if len(sessions) > 5:
        print(f"  ... and {len(sessions) - 5} more")
    
    # Confirm download
    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Download files
    download_sessions(sessions)
    
    # Verify downloads
    output_dir = 'downloaded_sessions'
    downloaded = list(Path(output_dir).glob('*.parquet'))
    print(f"\nVerification: {len(downloaded)} files in {output_dir}/")

if __name__ == "__main__":
    main()