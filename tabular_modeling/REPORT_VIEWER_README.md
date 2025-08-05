# ML Pipeline Report Viewer

## Overview
The ML pipeline now includes an automatic report server that makes it easy to view HTML reports when running on a remote server through VS Code.

## How It Works

### Automatic Server Start
When you run either `ml_pipeline_with_pcl.py` or `ml_pipeline_kfold.py`, the report server will automatically:
1. Start a web server on port 8888 (or next available port)
2. Display the URL to view the report
3. Provide instructions for VS Code port forwarding

### Manual Report Viewing
To view existing reports:

```bash
# List all available reports
python view_reports.py -l

# View a specific report (by number from list)
python view_reports.py 1

# View report on specific port
python view_reports.py 1 --port 9000
```

### VS Code Remote Setup

When the server starts, you'll see output like:
```
============================================================
📊 ML Report Server Started!
============================================================
Server running at: http://localhost:8888
Serving directory: /path/to/ml_output

To view reports:
  1. In VS Code: Forward port 8888 (Ports panel)
  2. Open in browser: http://localhost:8888
============================================================
```

To view reports in VS Code Remote:

1. **Open the Ports Panel**
   - View → Terminal → Ports
   - Or click "PORTS" tab in the terminal panel

2. **Forward the Port**
   - Click the "+" button in the Ports panel
   - Enter port number (e.g., 8888)
   - Or VS Code may auto-detect and offer to forward it

3. **Open the Report**
   - Click the forwarded port URL in the Ports panel
   - Or manually open http://localhost:8888 in your browser

## Disabling the Server

If you don't want the server to start automatically:
```bash
export NO_REPORT_SERVER=1
python ml_pipeline_with_pcl.py
```

## Troubleshooting

### Port Already in Use
The server will automatically find the next available port if 8888 is taken.

### Can't Access Reports
1. Ensure port forwarding is active in VS Code
2. Check that the server is running (look for "ML Report Server Started" message)
3. Try a different port: `python view_reports.py 1 --port 9999`

### Server Stops Immediately
The server runs in the background. If your script exits immediately, use:
```bash
python report_server.py  # Run server standalone
```