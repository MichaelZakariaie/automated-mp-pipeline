# Report Viewer Setup Complete ✅

## What's Been Implemented

### 1. **Automatic Report Server** (`report_server.py`)
- Starts automatically when ML pipelines generate reports
- Finds available ports if default (8888) is taken
- Serves all reports from `ml_output/` directory
- Shows recent reports on startup

### 2. **Enhanced Report Generator** (`report_generator.py`)
- Automatically starts report server after generating reports
- Displays clear instructions for VS Code port forwarding
- Shows direct URL to view the generated report

### 3. **Convenient Report Viewer** (`view_reports.py`)
- List all available reports: `python view_reports.py -l`
- View specific report: `python view_reports.py 1`
- Specify custom port: `python view_reports.py 1 --port 9000`

## Quick Start

### Running ML Pipeline
When you run either pipeline:
```bash
python ml_pipeline_with_pcl.py
# or
python ml_pipeline_kfold.py
```

You'll see:
```
============================================================
📊 View report at: http://localhost:8888/run_20250717_105952_ptsd_bin/report.html
============================================================

If using VS Code Remote:
  1. Open the Ports panel (View → Terminal → Ports)
  2. Forward port 8888
  3. Click the forwarded URL to view the report
============================================================
```

### VS Code Port Forwarding
1. **Method 1: Automatic**
   - VS Code often auto-detects and prompts to forward ports
   - Click "Yes" when prompted

2. **Method 2: Manual**
   - Open Command Palette (Ctrl/Cmd + Shift + P)
   - Type "Forward a Port"
   - Enter port number (8888)
   - Click the localhost URL that appears

3. **Method 3: Ports Panel**
   - View → Terminal → Ports
   - Click "+" button
   - Enter port 8888
   - Click the forwarded URL

### Viewing Existing Reports
```bash
# List all reports
python view_reports.py -l

# View report #1 from the list
python view_reports.py 1
```

## Environment Variables

- `NO_REPORT_SERVER=1` - Disable automatic server start
- `VSCODE_IPC_HOOK_CLI` - Auto-detected to show VS Code-specific instructions

## Files Created
- `report_server.py` - HTTP server for serving reports
- `view_reports.py` - CLI tool for viewing reports
- `test_report_server.py` - Test script for server functionality
- Updated `report_generator.py` - Integrated with report server

## Tested and Working ✅
- Server starts automatically with ML pipelines
- Port forwarding instructions displayed
- Multiple reports can be viewed
- Server handles port conflicts gracefully
- Works with both regular and k-fold pipelines