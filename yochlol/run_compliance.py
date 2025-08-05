import argparse
import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

# Configuration
PROCESSING_WORKERS = 24
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def find_videos(parent_dir):
    """Recursively finds video files under the parent directory"""
    return sorted(
        [
            str(p)
            for p in parent_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )


def has_existing_output(video_path, metrics_path):
    """Check if video already has JSON output file"""
    video_stem = Path(video_path).stem
    # Look for JSON files that match: {video_stem}_analysis_{date}_{time}.json
    pattern = os.path.join(metrics_path, f"{video_stem}_analysis_*.json")
    existing_files = glob.glob(pattern)
    return len(existing_files) > 0


def process_video(video_path, metrics_path):
    path = Path(video_path)
    try:
        cmd = [
            "python",
            "unified_compliance_analyzer.py",
            str(path),
            "--output-dir",
            str(metrics_path),
            "--ud",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            "filename": path.name,
            "processed_at": datetime.now().isoformat(),
            "subproc_status": "success",
            "file_size": path.stat().st_size,
            "error": None,
            # "stdout": result.stdout.strip(),
            "stdout": "masked",
            "stderr": result.stderr.strip(),
        }
    except subprocess.CalledProcessError as e:
        return {
            "filename": path.name,
            "processed_at": datetime.now().isoformat(),
            "subproc_status": "error",
            "file_size": path.stat().st_size,
            "error": str(e),
            "stdout": e.stdout.strip() if e.stdout else None,
            "stderr": e.stderr.strip() if e.stderr else None,
        }


def save_results_to_csv(results, output_file="video_processing_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main(args):
    video_dir = Path(args.video_dir)
    metrics_path = Path(args.metrics_path)

    if not args.dry_run:
        os.makedirs(metrics_path, exist_ok=True)

    video_paths = find_videos(video_dir)
    print(f"Found {len(video_paths)} videos.")

    if args.run_new:
        original_count = len(video_paths)
        video_paths = [
            path for path in video_paths if not has_existing_output(path, metrics_path)
        ]
        print(
            f"Filtering to new videos only: {len(video_paths)} new videos (skipped {original_count - len(video_paths)} already processed)."
        )

    if args.limit is not None:
        video_paths = video_paths[: args.limit]
        print(f"Limiting to first {args.limit} videos.")

    if args.dry_run:
        print(f"🔍 Dry run mode: Would process {len(video_paths)} videos.")
        for i, path in enumerate(video_paths, 1):
            print(f"  {i}. {Path(path).name}")
        return

    results = []

    with ProcessPoolExecutor(max_workers=PROCESSING_WORKERS) as executor:
        future_to_video = {
            executor.submit(process_video, path, metrics_path): path
            for path in video_paths
        }

        for future in as_completed(future_to_video):
            vid_path = future_to_video[future]
            try:
                result = future.result()
                result["video_path"] = vid_path
                results.append(result)
                print(f"✅ {Path(vid_path).name} [{result['subproc_status']}]")
            except Exception as e:
                print(f"❌ Error processing {vid_path}: {e}")
                results.append(
                    {
                        "filename": Path(vid_path).name,
                        "processed_at": datetime.now().isoformat(),
                        "subproc_status": "crash",
                        "error": str(e),
                    }
                )

    save_results_to_csv(results)
    print(f"✔️ All done. {len(results)} videos processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos for compliance.")
    parser.add_argument(
        "--limit", type=int, default=3, help="Maximum number of videos to process"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/media/m/mp_compliance/tmp/videos",
        help="Directory containing videos to process",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="/media/m/mp_compliance/tmp/metrics",
        help="Directory to save metrics output",
    )
    parser.add_argument(
        "--run-new",
        action="store_true",
        help="Only process videos that don't already have output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count and list videos that would be processed without actually processing them",
    )
    args = parser.parse_args()
    main(args)
