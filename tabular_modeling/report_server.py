#!/usr/bin/env python3
"""
Simple HTTP server for viewing ML pipeline HTML reports.
Automatically starts when reports are generated and provides easy viewing URLs.
"""

import http.server
import socketserver
import threading
import time
import webbrowser
import os
from pathlib import Path
import socket
import subprocess
import sys


class ReportServer:
    def __init__(self, port=8888, directory='ml_output'):
        self.port = port
        self.directory = Path(directory).absolute()
        self.server = None
        self.server_thread = None
        self.is_running = False
        
    def find_free_port(self):
        """Find a free port if the default is taken."""
        for port in range(self.port, self.port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except:
                    continue
        return None
        
    def start(self):
        """Start the HTTP server in a background thread."""
        if self.is_running:
            return self.port
            
        # Find available port
        actual_port = self.find_free_port()
        if not actual_port:
            print("No available ports found")
            return None
            
        self.port = actual_port
        
        # Change to the directory to serve
        os.chdir(self.directory)
        
        # Create server
        handler = http.server.SimpleHTTPRequestHandler
        self.server = socketserver.TCPServer(("", self.port), handler)
        
        # Start server in background thread
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.is_running = True
        
        # Print access information
        print(f"\n{'='*60}")
        print("📊 ML Report Server Started!")
        print(f"{'='*60}")
        print(f"Server running at: http://localhost:{self.port}")
        print(f"Serving directory: {self.directory}")
        print("\nTo view reports:")
        print(f"  1. In VS Code: Forward port {self.port} (Ports panel)")
        print(f"  2. Open in browser: http://localhost:{self.port}")
        print("\nRecent reports:")
        self._list_recent_reports()
        print(f"{'='*60}\n")
        
        return self.port
        
    def _list_recent_reports(self):
        """List recent report files."""
        reports = list(self.directory.rglob("report.html"))
        reports.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for i, report in enumerate(reports[:5]):
            rel_path = report.relative_to(self.directory)
            print(f"  - http://localhost:{self.port}/{rel_path}")
            if i >= 4:
                break
                
    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.is_running = False
            print("Server stopped")
            
    def open_report(self, report_path):
        """Open a specific report in the browser."""
        if not self.is_running:
            self.start()
            
        rel_path = Path(report_path).relative_to(self.directory)
        url = f"http://localhost:{self.port}/{rel_path}"
        
        print(f"\nOpening report: {url}")
        
        # Try to detect if we're in a remote VS Code session
        if os.environ.get('VSCODE_IPC_HOOK_CLI'):
            print("Detected VS Code remote session - please forward port manually")
        else:
            webbrowser.open(url)
            
        return url


# Global server instance
_server_instance = None


def get_report_server(port=8888, directory='ml_output'):
    """Get or create the global report server instance."""
    global _server_instance
    
    if _server_instance is None:
        _server_instance = ReportServer(port=port, directory=directory)
        _server_instance.start()
        
    return _server_instance


def view_report(report_path):
    """Convenience function to view a report."""
    server = get_report_server()
    return server.open_report(report_path)


if __name__ == "__main__":
    # If run directly, start server and keep alive
    server = get_report_server()
    
    try:
        print("\nPress Ctrl+C to stop the server...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()