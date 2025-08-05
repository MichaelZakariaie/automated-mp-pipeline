#!/usr/bin/env python3
"""
Convenience script to view ML pipeline HTML reports
"""

import argparse
from pathlib import Path
from report_server import get_report_server
import sys


def list_reports(directory='ml_output'):
    """List all available reports"""
    ml_dir = Path(directory)
    if not ml_dir.exists():
        print(f"Directory {ml_dir} does not exist")
        return []
    
    reports = list(ml_dir.rglob("report.html"))
    
    # Sort by run number (extract from run_X_timestamp format) or timestamp
    def get_sort_key(path):
        name = path.parent.name
        if name.startswith("run_"):
            parts = name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                # New format: run_X_timestamp
                return (1, int(parts[1]))
        # Old format or k-fold, sort by modification time
        return (0, -path.stat().st_mtime)
    
    reports.sort(key=get_sort_key, reverse=True)
    
    print("\nAvailable reports:")
    print("=" * 60)
    
    for i, report in enumerate(reports):
        rel_path = report.relative_to(ml_dir)
        run_name = report.parent.name
        print(f"{i+1}. {rel_path} ({run_name})")
    
    print("=" * 60)
    return reports


def main():
    parser = argparse.ArgumentParser(description='View ML pipeline HTML reports')
    parser.add_argument('report_number', nargs='?', type=int,
                        help='Report number to view (from list)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available reports')
    parser.add_argument('--port', '-p', type=int, default=8888,
                        help='Port to run server on (default: 8888)')
    
    args = parser.parse_args()
    
    reports = list_reports()
    
    if args.list or not args.report_number:
        if not reports:
            print("No reports found")
        return
    
    if args.report_number < 1 or args.report_number > len(reports):
        print(f"Invalid report number. Choose between 1 and {len(reports)}")
        return
    
    # Start server and show selected report
    server = get_report_server(port=args.port)
    report_path = reports[args.report_number - 1]
    rel_path = report_path.relative_to(Path('ml_output').absolute())
    
    print(f"\nOpening report: {rel_path}")
    server.open_report(report_path)
    
    # Keep server running
    print("\nPress Ctrl+C to stop the server...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()


if __name__ == "__main__":
    main()