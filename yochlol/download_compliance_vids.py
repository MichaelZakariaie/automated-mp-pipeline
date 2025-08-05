#!/usr/bin/env python3
"""
Download compliance videos from S3 using session IDs.

This script downloads videos for sessions identified by grab_messy_proto_sessions().
Session directories are named {UUID}_{session_timestamp}.
Video files are named {UUID}_{upload_timestamp}_{task}.mp4.

By default, it downloads specific video types (calibration_1, plr, face_pairs, mckinnon).
For PLR videos, it selects the first one based on unix timestamp.
For face_pairs, it specifically looks for face_pairs_task_1.

Usage:
    python download_compliance_vids.py                    # Download specific video types
    python download_compliance_vids.py --download-all     # Download all videos
    python download_compliance_vids.py --dry-run          # Show what would be downloaded
    python download_compliance_vids.py --sync             # Skip sessions with existing videos
    python download_compliance_vids.py --sync --dry-run   # Dry run with sync
"""

import argparse
import concurrent.futures
import os
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd

# Configuration
DOWNLOAD_WORKERS = 24
DOWNLOAD_BUFFER = 20

# AWS Configuration
BUCKET_NAME = "senseye-ptsd"
S3_PREFIX = "public/ptsd_ios/"
DOWNLOAD_PATH = "/media/m/mp_compliance/tmp/videos"

# Video types to download by default (can be overridden)
DEFAULT_VIDEO_TYPES = ["calibration_1", "plr", "face_pairs", "mckinnon"]


# Initialize S3 client (will use your AWS credentials from ~/.aws/credentials or environment)
s3_client = boto3.client("s3")

# Create queues
download_queue = queue.Queue(maxsize=DOWNLOAD_BUFFER)

# Ensure temp directory exists
os.makedirs(DOWNLOAD_PATH, exist_ok=True)


def get_session_videos_from_s3(bucket, session_id, video_types=None):
    """Get list of video files for a specific session from S3 bucket"""
    if video_types is None:
        video_types = DEFAULT_VIDEO_TYPES

    session_prefix = f"{S3_PREFIX}{session_id}/"
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=session_prefix)

        session_videos = []
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if Path(key).suffix.lower() in video_extensions:
                        session_videos.append(key)

        # Filter videos by type if specified
        if video_types:
            filtered_videos = []
            for video_type in video_types:
                if video_type == "plr":
                    # Find PLR videos and sort by unix timestamp
                    plr_videos = [v for v in session_videos if "plr" in v]
                    if plr_videos:
                        # Sort by unix timestamp extracted from filename
                        plr_videos.sort(key=lambda x: extract_unix_timestamp(x))
                        filtered_videos.append(plr_videos[0])
                elif video_type == "face_pairs":
                    # Find face_pairs_task_1 specifically
                    face_pairs_videos = [
                        v for v in session_videos if "face_pairs_task_1" in v
                    ]
                    if face_pairs_videos:
                        filtered_videos.append(face_pairs_videos[0])
                else:
                    # Find first video containing the video_type string
                    for video_key in session_videos:
                        if video_type in video_key:
                            filtered_videos.append(video_key)
                            break  # Only take first match for each type
            return filtered_videos
        else:
            return session_videos

    except Exception as e:
        print(f"Error accessing session {session_id}: {e}")
        return []


def extract_unix_timestamp(filename):
    """Extract unix timestamp from filename, return 0 if not found"""
    import re

    # Look for 13-digit unix timestamp (milliseconds)
    match = re.search(r"(\d{13})", filename)
    if match:
        return int(match.group(1))
    # Look for 10-digit unix timestamp (seconds)
    match = re.search(r"(\d{10})", filename)
    if match:
        return int(match.group(1))
    return 0


def grab_messy_proto_sessions():
    # There's a lot of extra junk in the AWS bucket, only get what we want
    print("Gathering MP session list...")
    QUERY = """
        SELECT DISTINCT session_id
        FROM data_quality.master_query_session_completion_check
        WHERE study_location = 'Messy Prototyping'
        AND problem = 'none'
        """
    DATABASE = "data_quality"
    df = wr.athena.read_sql_query(
        QUERY, database=DATABASE, ctas_approach=True, chunksize=True
    )
    df = pd.concat(df, axis=0)
    session_ids = df["session_id"].unique().tolist()
    return session_ids


def get_existing_session_ids():
    """Get session IDs that already have videos downloaded"""
    if not os.path.exists(DOWNLOAD_PATH):
        return set()

    existing_sessions = set()
    for filename in os.listdir(DOWNLOAD_PATH):
        if "_" in filename:
            # Extract UUID from filename (format: {uuid}_{upload_timestamp}_{task}.mp4)
            # We need to match this UUID to session directories
            uuid_part = filename.split("_")[0]
            existing_sessions.add(uuid_part)

    return existing_sessions


def get_all_session_videos(session_ids, download_all=False, sync=False):
    """Get video keys for all sessions"""
    video_types = None if download_all else DEFAULT_VIDEO_TYPES
    all_video_keys = []

    # Filter out sessions that already have videos if sync is enabled
    if sync:
        existing_uuids = get_existing_session_ids()
        original_count = len(session_ids)
        # Filter out sessions where the UUID part matches existing downloads
        session_ids = [
            sid for sid in session_ids if sid.split("_")[0] not in existing_uuids
        ]
        if existing_uuids:
            print(
                f"Sync mode: Skipping {original_count - len(session_ids)} sessions that already have videos"
            )

    print(f"Scanning {len(session_ids)} sessions...")
    for i, session_id in enumerate(session_ids):
        if i % 10 == 0:
            print(f"Scanned {i}/{len(session_ids)} sessions...")

        session_videos = get_session_videos_from_s3(
            BUCKET_NAME, session_id, video_types
        )
        all_video_keys.extend(session_videos)

    return all_video_keys


def download_worker():
    """Worker function to download videos from S3"""
    while True:
        s3_key = download_queue.get()
        if s3_key is None:  # Poison pill to stop worker
            break

        # Use original filename since it already contains session_id
        filename = Path(s3_key).name
        local_path = os.path.join(DOWNLOAD_PATH, filename)

        # Skip if file already exists
        if os.path.exists(local_path):
            print(f"Skipping {s3_key} (already exists)")
            download_queue.task_done()
            continue

        try:
            print(f"Downloading {s3_key}...")
            s3_client.download_file(BUCKET_NAME, s3_key, local_path)
            print(f"Downloaded {s3_key}")
        except Exception as e:
            print(f"Download failed for {s3_key}: {e}")
        finally:
            download_queue.task_done()


def main(download_all=False, dry_run=False, sync=False):
    print("Getting session IDs from grab_messy_proto_sessions()...")
    session_ids = grab_messy_proto_sessions()
    print(f"Found {len(session_ids)} sessions")

    print("Getting video list from S3...")
    video_keys = get_all_session_videos(
        session_ids, download_all=download_all, sync=sync
    )
    print(f"Found {len(video_keys)} videos to download")

    if dry_run:
        print(
            f"\nDRY RUN - Would download {len(video_keys)} videos from {len(session_ids)} sessions"
        )
        if not download_all:
            print(f"Video types: {', '.join(DEFAULT_VIDEO_TYPES)}")
        else:
            print("Video types: ALL")
        return

    if not video_keys:
        print("No videos found!")
        return

    # Start worker threads
    download_threads = []

    # Start download workers
    for i in range(DOWNLOAD_WORKERS):
        t = threading.Thread(target=download_worker, name=f"Downloader-{i}")
        t.start()
        download_threads.append(t)

    # Queue all video keys for download (they're already full S3 keys)
    for video_key in video_keys:
        download_queue.put(video_key)

    # Wait for all downloads to complete
    download_queue.join()

    # Stop download workers
    for _ in range(DOWNLOAD_WORKERS):
        download_queue.put(None)  # Poison pill

    for t in download_threads:
        t.join()

    print("Download complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download compliance videos from S3 using session IDs"
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all videos PER TASK for each session, this will take forever, don't do it. (default: only download specific video types)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Skip sessions that already have videos in the download directory",
    )
    args = parser.parse_args()

    main(download_all=args.download_all, dry_run=args.dry_run, sync=args.sync)
