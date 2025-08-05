#!/usr/bin/env python3
"""Test report server functionality"""

from report_server import ReportServer
import time
import requests

# Start server
server = ReportServer(port=8889)  # Use different port for testing
port = server.start()

if port:
    print(f"✓ Server started successfully on port {port}")
    
    # Test if server is accessible
    try:
        response = requests.get(f"http://localhost:{port}", timeout=2)
        print(f"✓ Server is accessible (status: {response.status_code})")
    except Exception as e:
        print(f"✗ Server not accessible: {e}")
    
    # Stop server
    server.stop()
    print("✓ Server stopped successfully")
else:
    print("✗ Failed to start server")