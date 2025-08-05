#!/usr/bin/env python3
"""
Persistent run counter for ML pipeline outputs
"""

import json
from pathlib import Path
import fcntl
import os

class RunCounter:
    """Manages a persistent counter for run numbering"""
    
    def __init__(self, counter_file='.ml_run_counter.json'):
        self.counter_file = Path(counter_file)
        self.counter_data = self._load_counter()
    
    def _load_counter(self):
        """Load counter from file or initialize if not exists"""
        if self.counter_file.exists():
            try:
                with open(self.counter_file, 'r') as f:
                    return json.load(f)
            except:
                return {'next_run_number': 1}
        return {'next_run_number': 1}
    
    def _save_counter(self):
        """Save counter to file with file locking to prevent race conditions"""
        # Create temporary file
        temp_file = self.counter_file.with_suffix('.tmp')
        
        # Write to temporary file
        with open(temp_file, 'w') as f:
            json.dump(self.counter_data, f, indent=2)
        
        # Atomically replace the original file
        temp_file.replace(self.counter_file)
    
    def get_next_run_number(self):
        """Get the next run number and increment counter"""
        # Use file locking to ensure thread safety
        lock_file = self.counter_file.with_suffix('.lock')
        
        # Create lock file if it doesn't exist
        lock_file.touch()
        
        with open(lock_file, 'r+') as lock_fd:
            # Acquire exclusive lock
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            
            try:
                # Reload counter in case it was modified
                self.counter_data = self._load_counter()
                
                # Get current number
                run_number = self.counter_data['next_run_number']
                
                # Increment for next time
                self.counter_data['next_run_number'] = run_number + 1
                
                # Save updated counter
                self._save_counter()
                
                return run_number
                
            finally:
                # Release lock
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    
    def get_current_run_number(self):
        """Get the current run number without incrementing"""
        return self.counter_data.get('next_run_number', 1) - 1

def get_next_run_folder(base_dir='ml_output', timestamp_format='%Y%m%d_%H%M%S'):
    """Get the next run folder name with format: run_X_YYYYMMDD_HHMMSS"""
    from datetime import datetime
    
    counter = RunCounter()
    run_number = counter.get_next_run_number()
    timestamp = datetime.now().strftime(timestamp_format)
    
    folder_name = f"run_{run_number}_{timestamp}"
    full_path = Path(base_dir) / folder_name
    
    return full_path, run_number

if __name__ == "__main__":
    # Test the counter
    print("Testing run counter...")
    
    for i in range(3):
        folder, num = get_next_run_folder()
        print(f"Run {num}: {folder}")