#!/usr/bin/env python3
"""
Computer Vision Data Generator
Generates dummy time series data in the same format that would be produced by CV algorithms
This allows testing the pipeline before implementing actual CV processing
"""

import numpy as np
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import json
import random


class ComputerVisionDataGenerator:
    """Generate dummy CV time series data matching expected format"""
    
    def __init__(self, output_dir='cv_output', sessions_count=10, 
                 trials_per_session=140, fps=30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_count = sessions_count
        self.trials_per_session = trials_per_session
        self.fps = fps
        
        # Define data schemas matching yochlol expectations
        self.time_series_features = [
            'gaze_x', 'gaze_y',  # Gaze coordinates
            'pupil_diameter_left', 'pupil_diameter_right',  # Pupil data
            'face_confidence',  # Face detection confidence
            'emotion_valence', 'emotion_arousal',  # Emotion scores
            'head_pose_x', 'head_pose_y', 'head_pose_z',  # Head pose
            'blink_detected',  # Binary blink detection
            'saccade_amplitude', 'saccade_velocity',  # Eye movement
        ]
        
        # Features that match tabular modeling expectations
        self.trial_features = [
            'dot_latency_pog',
            'cue_latency_pog', 
            'trial_saccade_data_quality_pog',
            'cue_latency_pog_good',
            'percent_bottom_freeface_pog',
            'percent_top_freeface_pog',
            'fixation_quality'
        ]
        
    def generate_session_id(self):
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    def generate_time_series_data(self, duration_seconds=300):
        """Generate time series data for a single session"""
        n_frames = int(duration_seconds * self.fps)
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=random.randint(1, 30)),
            periods=n_frames,
            freq=f'{1000/self.fps}ms'
        )
        
        data = {
            'timestamp': timestamps,
            'frame_number': range(n_frames)
        }
        
        # Generate realistic-looking time series data
        for feature in self.time_series_features:
            if feature == 'blink_detected':
                # Binary feature with occasional blinks
                data[feature] = np.random.choice([0, 1], n_frames, p=[0.95, 0.05])
            elif 'confidence' in feature:
                # Confidence scores between 0 and 1
                data[feature] = np.random.beta(8, 2, n_frames)
            elif 'emotion' in feature:
                # Emotion scores with some temporal correlation
                base = np.random.randn(n_frames)
                smoothed = pd.Series(base).rolling(window=30, center=True).mean().fillna(0)
                data[feature] = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
            elif 'gaze' in feature:
                # Gaze coordinates with smooth movements
                base = np.cumsum(np.random.randn(n_frames) * 0.01)
                data[feature] = np.clip(base, -1, 1)
            elif 'pupil' in feature:
                # Pupil diameter with realistic range (2-8mm)
                data[feature] = 5 + np.random.randn(n_frames) * 0.5
                data[feature] = np.clip(data[feature], 2, 8)
            else:
                # Default: normally distributed data
                data[feature] = np.random.randn(n_frames)
        
        return pd.DataFrame(data)
    
    def generate_trial_data(self, session_id):
        """Generate trial-level data matching tabular modeling format"""
        trials = []
        
        for trial_num in range(self.trials_per_session):
            trial = {
                'session_id': session_id,
                'trial': trial_num,
                'dot_latency_pog': np.random.exponential(0.5) if random.random() > 0.1 else np.nan,
                'cue_latency_pog': np.random.exponential(0.3) if random.random() > 0.05 else np.nan,
                'trial_saccade_data_quality_pog': np.random.choice(['good', 'bad'], p=[0.8, 0.2]),
                'cue_latency_pog_good': np.random.choice(['good', 'bad'], p=[0.9, 0.1]),
                'percent_bottom_freeface_pog': np.random.beta(2, 2),
                'percent_top_freeface_pog': np.random.beta(2, 2),
                'fixation_quality': np.random.choice(['good', 'bad'], p=[0.85, 0.15])
            }
            
            # Ensure percentages sum to reasonable amount
            total_percent = trial['percent_bottom_freeface_pog'] + trial['percent_top_freeface_pog']
            if total_percent > 1:
                trial['percent_bottom_freeface_pog'] /= total_percent
                trial['percent_top_freeface_pog'] /= total_percent
                
            trials.append(trial)
        
        # Add session-level features
        df = pd.DataFrame(trials)
        df['percent_loss_late'] = np.random.beta(1, 10)  # Usually low
        df['session_saccade_data_quality_pog'] = np.random.choice(['good', 'bad'], p=[0.75, 0.25])
        
        return df
    
    def generate_compliance_metrics(self, session_id):
        """Generate compliance metrics matching yochlol format"""
        metrics = {
            'session_id': session_id,
            'compliance_score': np.random.beta(8, 2),  # Usually high
            'valid_trials': np.random.randint(100, 140),
            'calibration_quality': np.random.choice(['good', 'moderate', 'poor'], p=[0.7, 0.2, 0.1]),
            'tracking_loss_events': np.random.poisson(2),
            'average_gaze_accuracy': np.random.beta(9, 1),
            'head_movement_excessive': np.random.choice([True, False], p=[0.1, 0.9])
        }
        
        return metrics
    
    def save_time_series_data(self, data, session_id, data_type='raw'):
        """Save time series data in parquet format"""
        filename = f"{session_id}_{int(datetime.now().timestamp() * 1000)}_{data_type}.parquet"
        filepath = self.output_dir / 'time_series' / filename
        filepath.parent.mkdir(exist_ok=True)
        
        data.to_parquet(filepath, index=False)
        return filepath
    
    def save_trial_data(self, data, session_id):
        """Save trial data in format expected by tabular modeling"""
        filename = f"{session_id}_{int(datetime.now().timestamp() * 1000)}_inter_pir_face_pairs_v2_latedwell_pog.parquet"
        filepath = self.output_dir / 'trials' / filename
        filepath.parent.mkdir(exist_ok=True)
        
        data.to_parquet(filepath, index=False)
        return filepath
    
    def save_compliance_metrics(self, metrics, session_id):
        """Save compliance metrics as JSON"""
        filename = f"{session_id}_compliance_metrics.json"
        filepath = self.output_dir / 'compliance' / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return filepath
    
    def generate_session(self):
        """Generate all data for a single session"""
        session_id = self.generate_session_id()
        
        print(f"Generating data for session {session_id}")
        
        # Generate time series data (what CV would produce)
        time_series = self.generate_time_series_data()
        ts_path = self.save_time_series_data(time_series, session_id)
        
        # Generate trial-level aggregated data
        trial_data = self.generate_trial_data(session_id)
        trial_path = self.save_trial_data(trial_data, session_id)
        
        # Generate compliance metrics
        compliance = self.generate_compliance_metrics(session_id)
        compliance_path = self.save_compliance_metrics(compliance, session_id)
        
        return {
            'session_id': session_id,
            'time_series_path': str(ts_path),
            'trial_data_path': str(trial_path),
            'compliance_path': str(compliance_path)
        }
    
    def generate_all_sessions(self):
        """Generate data for all sessions"""
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'sessions_count': self.sessions_count,
            'fps': self.fps,
            'trials_per_session': self.trials_per_session,
            'sessions': []
        }
        
        for i in range(self.sessions_count):
            print(f"Generating session {i+1}/{self.sessions_count}")
            session_info = self.generate_session()
            manifest['sessions'].append(session_info)
        
        # Save manifest
        manifest_path = self.output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Generated {self.sessions_count} sessions")
        print(f"Data saved to: {self.output_dir}")
        print(f"Manifest saved to: {manifest_path}")
        
        return manifest


def main():
    """Standalone execution for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dummy CV data')
    parser.add_argument('--output-dir', default='cv_output',
                        help='Output directory for generated data')
    parser.add_argument('--sessions', type=int, default=5,
                        help='Number of sessions to generate')
    parser.add_argument('--trials', type=int, default=140,
                        help='Trials per session')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for time series data')
    
    args = parser.parse_args()
    
    generator = ComputerVisionDataGenerator(
        output_dir=args.output_dir,
        sessions_count=args.sessions,
        trials_per_session=args.trials,
        fps=args.fps
    )
    
    generator.generate_all_sessions()


if __name__ == '__main__':
    main()