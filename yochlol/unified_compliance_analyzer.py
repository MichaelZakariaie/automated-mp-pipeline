#!/usr/bin/env python3
"""
Unified Compliance Analysis Script

This script provides a comprehensive analysis framework for video compliance monitoring.
It integrates multiple analysis algorithms and provides extensible architecture for future additions.

Current Algorithms:
1. Basic Video Analysis (frame count, duration, etc.)
2. MediaPipe Face Mesh IMU Analysis (orientation, depth, velocity)

Framework for Future Algorithms:
3. Eye Tracking Analysis [PLACEHOLDER]
4. Facial Expression Analysis [PLACEHOLDER]
5. Attention/Focus Analysis [PLACEHOLDER]
6. Behavioral Pattern Analysis [PLACEHOLDER]
7. Compliance Score Calculation [PLACEHOLDER]
8. Anomaly Detection [PLACEHOLDER]

Usage:
    python unified_compliance_analyzer.py <video_path> [options]
"""

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import seaborn as sns


class AnalysisAlgorithm(ABC):
    """Abstract base class for analysis algorithms"""

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the algorithm"""
        pass

    @abstractmethod
    def initialize(self, video_path: str, **kwargs) -> bool:
        """Initialize the algorithm with video parameters"""
        pass

    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict[str, Any]:
        """Process a single frame and return analysis data"""
        pass

    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """Finalize processing and return summary statistics"""
        pass

    @abstractmethod
    def get_column_headers(self) -> List[str]:
        """Return column headers for CSV output"""
        pass


class BasicVideoAnalysis(AnalysisAlgorithm):
    """Basic video analysis including frame count, duration, resolution, etc."""

    def __init__(self):
        self.frame_count = 0
        self.video_path = ""
        self.cap = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.duration = 0

    def get_name(self) -> str:
        return "BasicVideoAnalysis"

    def initialize(self, video_path: str, **kwargs) -> bool:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        return True

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict[str, Any]:
        self.frame_count += 1

        # Only track brightness
        brightness = np.mean(frame)

        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "brightness": brightness,
        }

    def finalize(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "processed_frames": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration,
            "resolution": f"{self.width}x{self.height}",
            "processing_completion_rate": (
                (self.frame_count / self.total_frames * 100)
                if self.total_frames > 0
                else 0
            ),
        }

    def get_column_headers(self) -> List[str]:
        return ["frame_number", "timestamp", "brightness"]


class FaceMeshIMUAnalysis(AnalysisAlgorithm):
    """MediaPipe Face Mesh IMU Analysis with orientation, depth, and velocity tracking"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.model_points = None
        self.landmark_indices = None
        self.prev_roll = None
        self.prev_pitch = None
        self.prev_yaw = None
        self.prev_timestamp = None
        self.frame_data = []

    def get_name(self) -> str:
        return "FaceMeshIMUAnalysis"

    def initialize(self, video_path: str, **kwargs) -> bool:
        # Initialize MediaPipe Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Get video dimensions for camera matrix
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Camera parameters
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
        self.dist_coeffs = np.zeros((4, 1))

        # 3D model points
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
            ]
        )

        self.landmark_indices = [1, 152, 263, 33, 287, 57]

        return True

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict[str, Any]:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        # Default values
        data = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "face_detected": False,
            "roll": np.nan,
            "pitch": np.nan,
            "yaw": np.nan,
            "x_position": np.nan,
            "y_position": np.nan,
            "z_depth": np.nan,
            "roll_velocity": np.nan,
            "pitch_velocity": np.nan,
            "yaw_velocity": np.nan,
            "total_velocity": np.nan,
            "face_size_percentage": np.nan,
            "face_bbox_width": np.nan,
            "face_bbox_height": np.nan,
        }

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            data["face_detected"] = True

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height

            # Calculate face bounding box from all landmarks
            all_x_coords = []
            all_y_coords = []

            for landmark in face_landmarks.landmark:
                x = landmark.x * frame_width
                y = landmark.y * frame_height
                all_x_coords.append(x)
                all_y_coords.append(y)

            # Calculate bounding box
            min_x, max_x = min(all_x_coords), max(all_x_coords)
            min_y, max_y = min(all_y_coords), max(all_y_coords)

            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            bbox_area = bbox_width * bbox_height

            # Calculate face size as percentage of screen
            face_size_percentage = bbox_area / frame_area

            # Update face size data
            data.update(
                {
                    "face_size_percentage": face_size_percentage,
                    "face_bbox_width": bbox_width,
                    "face_bbox_height": bbox_height,
                }
            )

            # Prepare image points for pose estimation
            image_points = []

            for idx in self.landmark_indices:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x = int(lm.x * frame_width)
                    y = int(lm.y * frame_height)
                    image_points.append((x, y))

            if len(image_points) == len(self.model_points):
                image_points = np.array(image_points, dtype="double")

                # Solve PnP
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                if success:
                    # Convert to Euler angles
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    translation_vector = translation_vector.reshape(3, 1)
                    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
                    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(
                        pose_matrix
                    )
                    yaw, pitch, roll = eulerAngles.flatten()

                    # Calculate velocities
                    roll_velocity = pitch_velocity = yaw_velocity = 0.0
                    if self.prev_roll is not None and self.prev_timestamp is not None:
                        dt = timestamp - self.prev_timestamp
                        if dt > 0:
                            roll_velocity = (roll - self.prev_roll) / dt
                            pitch_velocity = (pitch - self.prev_pitch) / dt
                            yaw_velocity = (yaw - self.prev_yaw) / dt

                    # Update data
                    data.update(
                        {
                            "roll": roll,
                            "pitch": pitch,
                            "yaw": yaw,
                            "x_position": translation_vector[0][0],
                            "y_position": translation_vector[1][0],
                            "z_depth": translation_vector[2][0],
                            "roll_velocity": roll_velocity,
                            "pitch_velocity": pitch_velocity,
                            "yaw_velocity": yaw_velocity,
                            "total_velocity": np.sqrt(
                                roll_velocity**2 + pitch_velocity**2 + yaw_velocity**2
                            ),
                        }
                    )

                    # Update previous values
                    self.prev_roll, self.prev_pitch, self.prev_yaw = roll, pitch, yaw
                    self.prev_timestamp = timestamp

        self.frame_data.append(data)
        return data

    def finalize(self) -> Dict[str, Any]:
        if not self.frame_data:
            return {"error": "No data processed"}

        df = pd.DataFrame(self.frame_data)
        df_valid = df[df["face_detected"] == True]

        if len(df_valid) == 0:
            return {"error": "No valid face detections"}

        # Calculate comprehensive statistics
        stats = {
            "total_frames": len(df),
            "valid_detections": len(df_valid),
            "detection_rate": len(df_valid) / len(df) * 100,
            # Orientation statistics
            "orientation_stats": {
                "roll": self._calculate_stats(df_valid["roll"]),
                "pitch": self._calculate_stats(df_valid["pitch"]),
                "yaw": self._calculate_stats(df_valid["yaw"]),
            },
            # Position statistics
            "position_stats": {
                "x": self._calculate_stats(df_valid["x_position"]),
                "y": self._calculate_stats(df_valid["y_position"]),
                "z_depth": self._calculate_stats(df_valid["z_depth"]),
            },
            # Velocity statistics
            "velocity_stats": {
                "roll": self._calculate_stats(df_valid["roll_velocity"]),
                "pitch": self._calculate_stats(df_valid["pitch_velocity"]),
                "yaw": self._calculate_stats(df_valid["yaw_velocity"]),
                "total": self._calculate_stats(df_valid["total_velocity"]),
            },
            # Movement analysis
            "movement_analysis": {
                "high_movement_frames": int((df_valid["total_velocity"] > 100).sum()),
                "moderate_movement_frames": int(
                    (
                        (df_valid["total_velocity"] >= 20)
                        & (df_valid["total_velocity"] <= 100)
                    ).sum()
                ),
                "low_movement_frames": int((df_valid["total_velocity"] < 20).sum()),
            },
            # Face size analysis
            "face_size_stats": {
                "percentage": self._calculate_stats(df_valid["face_size_percentage"]),
                "bbox_width": self._calculate_stats(df_valid["face_bbox_width"]),
                "bbox_height": self._calculate_stats(df_valid["face_bbox_height"]),
            },
            # Compliance analysis
            "compliance_analysis": {
                "optimal_size_frames": int(
                    (
                        (df_valid["face_size_percentage"] >= 0.05)
                        & (df_valid["face_size_percentage"] <= 0.25)
                    ).sum()
                ),
                "too_small_frames": int(
                    (df_valid["face_size_percentage"] < 0.05).sum()
                ),
                "too_large_frames": int(
                    (df_valid["face_size_percentage"] > 0.25).sum()
                ),
                "optimal_size_percentage": float(
                    (
                        (df_valid["face_size_percentage"] >= 0.05)
                        & (df_valid["face_size_percentage"] <= 0.25)
                    ).sum()
                    / len(df_valid)
                    * 100
                ),
            },
        }

        return stats

    def _calculate_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive statistics for a data series"""
        return {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "range": float(series.max() - series.min()),
        }

    def get_column_headers(self) -> List[str]:
        return [
            "frame_number",
            "timestamp",
            "face_detected",
            "roll",
            "pitch",
            "yaw",
            "x_position",
            "y_position",
            "z_depth",
            "roll_velocity",
            "pitch_velocity",
            "yaw_velocity",
            "total_velocity",
            "face_size_percentage",
            "face_bbox_width",
            "face_bbox_height",
        ]


# Placeholder classes for future algorithms
class EyeTrackingAnalysis(AnalysisAlgorithm):
    """Placeholder for eye tracking analysis"""

    def get_name(self) -> str:
        return "EyeTrackingAnalysis"

    def initialize(self, video_path: str, **kwargs) -> bool:
        print(f"[{self.get_name()}] PLACEHOLDER - Not yet implemented")
        return False

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict[str, Any]:
        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "placeholder": True,
        }

    def finalize(self) -> Dict[str, Any]:
        return {"status": "Not implemented"}

    def get_column_headers(self) -> List[str]:
        return ["frame_number", "timestamp", "placeholder"]


class FacialExpressionAnalysis(AnalysisAlgorithm):
    """Placeholder for facial expression analysis"""

    def get_name(self) -> str:
        return "FacialExpressionAnalysis"

    def initialize(self, video_path: str, **kwargs) -> bool:
        print(f"[{self.get_name()}] PLACEHOLDER - Not yet implemented")
        return False

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict[str, Any]:
        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "placeholder": True,
        }

    def finalize(self) -> Dict[str, Any]:
        return {"status": "Not implemented"}

    def get_column_headers(self) -> List[str]:
        return ["frame_number", "timestamp", "placeholder"]


class AttentionFocusAnalysis(AnalysisAlgorithm):
    """Placeholder for attention/focus analysis"""

    def get_name(self) -> str:
        return "AttentionFocusAnalysis"

    def initialize(self, video_path: str, **kwargs) -> bool:
        print(f"[{self.get_name()}] PLACEHOLDER - Not yet implemented")
        return False

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict[str, Any]:
        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "placeholder": True,
        }

    def finalize(self) -> Dict[str, Any]:
        return {"status": "Not implemented"}

    def get_column_headers(self) -> List[str]:
        return ["frame_number", "timestamp", "placeholder"]


class ComplianceScoreCalculator(AnalysisAlgorithm):
    """Placeholder for compliance score calculation"""

    def get_name(self) -> str:
        return "ComplianceScoreCalculator"

    def initialize(self, video_path: str, **kwargs) -> bool:
        print(f"[{self.get_name()}] PLACEHOLDER - Not yet implemented")
        return False

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict[str, Any]:
        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "placeholder": True,
        }

    def finalize(self) -> Dict[str, Any]:
        return {"status": "Not implemented"}

    def get_column_headers(self) -> List[str]:
        return ["frame_number", "timestamp", "placeholder"]


class UnifiedComplianceAnalyzer:
    """Main analyzer class that orchestrates all analysis algorithms"""

    def __init__(self):
        # Initialize all algorithms
        self.algorithms = [
            BasicVideoAnalysis(),
            FaceMeshIMUAnalysis(),
            # Placeholder algorithms (will be skipped if not implemented)
            EyeTrackingAnalysis(),
            FacialExpressionAnalysis(),
            AttentionFocusAnalysis(),
            ComplianceScoreCalculator(),
        ]

        self.active_algorithms = []
        self.video_path = ""
        self.output_dir = ""
        self.frame_data = []

    def analyze_video(
        self,
        video_path: str,
        output_dir: str = None,
        enabled_algorithms: List[str] = None,
        upside_down: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze a video using all enabled algorithms

        Args:
            video_path: Path to the video file
            output_dir: Directory to save outputs (default: same as video)
            enabled_algorithms: List of algorithm names to enable (default: all implemented)

        Returns:
            Dict containing all analysis results
        """

        self.video_path = video_path
        self.output_dir = output_dir or os.path.dirname(video_path)

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Initialize algorithms
        print("Initializing algorithms...")
        for algo in self.algorithms:
            if enabled_algorithms is None or algo.get_name() in enabled_algorithms:
                if algo.initialize(video_path):
                    self.active_algorithms.append(algo)
                    print(f"✓ {algo.get_name()} initialized")
                else:
                    print(f"✗ {algo.get_name()} failed to initialize")

        if not self.active_algorithms:
            raise ValueError("No algorithms successfully initialized")

        # Process video
        print(f"\nProcessing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_number = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Do a barrel roll!
            if upside_down:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame_number += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Process frame with all active algorithms
            frame_results = {}
            for algo in self.active_algorithms:
                try:
                    result = algo.process_frame(frame, frame_number, timestamp)
                    frame_results[algo.get_name()] = result
                except Exception as e:
                    print(f"Error in {algo.get_name()}: {e}")
                    frame_results[algo.get_name()] = {"error": str(e)}

            self.frame_data.append(frame_results)

            # Progress indicator
            if frame_number % 50 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames})")

        cap.release()

        # Finalize algorithms and collect results
        print("\nFinalizing analysis...")
        final_results = {
            "metadata": {
                "video_path": video_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_frames_processed": frame_number,
                "active_algorithms": [
                    algo.get_name() for algo in self.active_algorithms
                ],
            },
            "algorithm_results": {},
            "frame_data": self.frame_data,
        }

        for algo in self.active_algorithms:
            try:
                final_results["algorithm_results"][algo.get_name()] = algo.finalize()
                print(f"✓ {algo.get_name()} finalized")
            except Exception as e:
                print(f"✗ Error finalizing {algo.get_name()}: {e}")
                final_results["algorithm_results"][algo.get_name()] = {"error": str(e)}

        # Save results
        self._save_results(final_results)

        return final_results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to files"""

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = os.path.join(
            self.output_dir, f"{base_name}_analysis_{timestamp}.json"
        )
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"JSON results saved to: {json_file}")

        # Save unified CSV data with all algorithms
        try:
            # Create a list to store all rows with combined data
            unified_data = []

            # Process each frame's data
            for frame_idx, frame_data in enumerate(self.frame_data):
                # Start with an empty row for this frame
                row = {}

                # Add data from each algorithm to the row
                for algo_name in [a.get_name() for a in self.active_algorithms]:
                    if algo_name in frame_data:
                        algo_result = frame_data[algo_name]

                        # Skip if there's an error
                        if isinstance(algo_result, dict) and "error" not in algo_result:
                            # Prefix each column with the algorithm name
                            for key, value in algo_result.items():
                                # For frame_number and timestamp, only include once (not prefixed)
                                if key in ["frame_number", "timestamp"] and key in row:
                                    continue
                                elif key in ["frame_number", "timestamp"]:
                                    row[key] = value
                                else:
                                    # Prefix other columns with algorithm name
                                    row[f"{algo_name}_{key}"] = value

                # Only add row if it has data
                if row:
                    unified_data.append(row)

            # Create DataFrame and save unified CSV
            if unified_data:
                df = pd.DataFrame(unified_data)

                # Reorder columns to have frame_number and timestamp first
                priority_cols = ["frame_number", "timestamp"]
                other_cols = [col for col in df.columns if col not in priority_cols]
                ordered_cols = priority_cols + sorted(other_cols)
                df = df[ordered_cols]

                # Save the unified CSV
                csv_file = os.path.join(
                    self.output_dir, f"{base_name}_unified_analysis_{timestamp}.csv"
                )
                df.to_csv(csv_file, index=False)
                print(f"Unified CSV saved to: {csv_file}")
                print(f"Total columns: {len(df.columns)}")
                print(f"Total rows: {len(df)}")
            else:
                print("No data to save to CSV")

        except Exception as e:
            print(f"Error saving unified CSV: {e}")

        # Generate summary report
        self._generate_summary_report(results, base_name, timestamp)

    def _generate_summary_report(
        self, results: Dict[str, Any], base_name: str, timestamp: str
    ) -> None:
        """Generate a human-readable summary report"""

        report_file = os.path.join(
            self.output_dir, f"{base_name}_summary_{timestamp}.txt"
        )

        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("UNIFIED COMPLIANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            f.write("ANALYSIS METADATA:\n")
            f.write(f"Video: {results['metadata']['video_path']}\n")
            f.write(f"Analysis Time: {results['metadata']['analysis_timestamp']}\n")
            f.write(
                f"Frames Processed: {results['metadata']['total_frames_processed']}\n"
            )
            f.write(
                f"Active Algorithms: {', '.join(results['metadata']['active_algorithms'])}\n\n"
            )

            # Algorithm results
            for algo_name, algo_results in results["algorithm_results"].items():
                f.write(f"{algo_name.upper()} RESULTS:\n")
                f.write("-" * 50 + "\n")

                if "error" in algo_results:
                    f.write(f"Error: {algo_results['error']}\n\n")
                    continue

                # Custom formatting based on algorithm type
                if algo_name == "BasicVideoAnalysis":
                    f.write(f"Resolution: {algo_results.get('resolution', 'N/A')}\n")
                    f.write(f"FPS: {algo_results.get('fps', 'N/A'):.1f}\n")
                    f.write(
                        f"Duration: {algo_results.get('duration_seconds', 'N/A'):.2f} seconds\n"
                    )
                    f.write(
                        f"Processing Rate: {algo_results.get('processing_completion_rate', 'N/A'):.1f}%\n"
                    )

                elif algo_name == "FaceMeshIMUAnalysis":
                    f.write(
                        f"Face Detection Rate: {algo_results.get('detection_rate', 'N/A'):.1f}%\n"
                    )
                    f.write(
                        f"Valid Detections: {algo_results.get('valid_detections', 'N/A')}\n"
                    )

                    # Orientation stats
                    if "orientation_stats" in algo_results:
                        f.write("\nOrientation Statistics:\n")
                        for axis, stats in algo_results["orientation_stats"].items():
                            f.write(
                                f"  {axis.capitalize()}: {stats['mean']:.2f}° ± {stats['std']:.2f}°\n"
                            )

                    # Movement analysis
                    if "movement_analysis" in algo_results:
                        f.write("\nMovement Analysis:\n")
                        ma = algo_results["movement_analysis"]
                        f.write(
                            f"  High Movement Frames: {ma.get('high_movement_frames', 0)}\n"
                        )
                        f.write(
                            f"  Moderate Movement Frames: {ma.get('moderate_movement_frames', 0)}\n"
                        )
                        f.write(
                            f"  Low Movement Frames: {ma.get('low_movement_frames', 0)}\n"
                        )

                    # Face size analysis
                    if "face_size_stats" in algo_results:
                        f.write("\nFace Size Analysis:\n")
                        fs = algo_results["face_size_stats"]["percentage"]
                        f.write(
                            f"  Average Face Size: {fs['mean']*100:.2f}% of screen\n"
                        )
                        f.write(
                            f"  Face Size Range: {fs['min']*100:.2f}% - {fs['max']*100:.2f}%\n"
                        )
                        f.write(f"  Face Size Std Dev: ±{fs['std']*100:.2f}%\n")

                    # Compliance analysis
                    if "compliance_analysis" in algo_results:
                        f.write("\nCompliance Analysis (Face Size):\n")
                        ca = algo_results["compliance_analysis"]
                        f.write(
                            f"  Optimal Size Frames (5-25%): {ca.get('optimal_size_frames', 0)} ({ca.get('optimal_size_percentage', 0):.1f}%)\n"
                        )
                        f.write(
                            f"  Too Small Frames (<5%): {ca.get('too_small_frames', 0)}\n"
                        )
                        f.write(
                            f"  Too Large Frames (>25%): {ca.get('too_large_frames', 0)}\n"
                        )

                else:
                    # Generic formatting for other algorithms
                    f.write(f"Status: {algo_results.get('status', 'Unknown')}\n")

                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"Summary report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Unified Compliance Analysis Tool")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="Specific algorithms to run (default: all available)",
    )
    parser.add_argument(
        "--list-algorithms", action="store_true", help="List all available algorithms"
    )
    parser.add_argument(
        "--ud",
        action="store_true",
        help="Process video upside down (rotate 180 degrees)",
    )

    args = parser.parse_args()

    if args.list_algorithms:
        analyzer = UnifiedComplianceAnalyzer()
        print("Available algorithms:")
        for algo in analyzer.algorithms:
            print(f"  - {algo.get_name()}")
        return

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    try:
        analyzer = UnifiedComplianceAnalyzer()
        results = analyzer.analyze_video(
            args.video_path, args.output_dir, args.algorithms, args.ud
        )

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"Processed {results['metadata']['total_frames_processed']} frames")
        print(f"Used {len(results['metadata']['active_algorithms'])} algorithms")

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
