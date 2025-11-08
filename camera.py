import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
import time

@dataclass
class PostureMetrics:
    """Data class to store posture analysis metrics"""
    shoulder_angle: float  # Angle of shoulders (should be ~0 for level)
    head_tilt: float  # Head tilt angle
    forward_lean: float  # How far forward the head is from shoulders
    is_slouching: bool
    is_tilted: bool
    is_leaning: bool
    timestamp: float
    issues: List[str]

class PostureDetector:
    """Detects and analyzes posture using MediaPipe Pose"""
    
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector with optimized parameters
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # 0, 1, or 2 (higher = more accurate but slower)
        )
        
        # Thresholds for posture issues (these can be tuned)
        self.FORWARD_LEAN_THRESHOLD = 0.15  # ratio of head-to-shoulder distance
        self.HEAD_TILT_MIN = 165
        self.HEAD_TILT_MAX = 195
        self.SHOULDER_ANGLE_MIN = 165
        self.SHOULDER_ANGLE_MAX = 195

        # For smoothing measurements
        self.metric_history = {
            'shoulder_angle': [],
            'head_tilt': [],
            'forward_lean': []
        }
        self.history_size = 5  # Number of frames to average
        
    def calculate_angle(self, point1: tuple, point2: tuple, point3: tuple) -> float:
        """
        Calculate angle between three points
        point2 is the vertex of the angle
        """
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_slope_angle(self, point1: tuple, point2: tuple) -> float:
        """Calculate the angle of a line from horizontal"""
        x1, y1 = point1
        x2, y2 = point2
        
        # Calculate angle from horizontal
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return abs(angle)
    
    def smooth_metric(self, metric_name: str, value: float) -> float:
        """Apply moving average smoothing to reduce jitter"""
        self.metric_history[metric_name].append(value)
        
        if len(self.metric_history[metric_name]) > self.history_size:
            self.metric_history[metric_name].pop(0)
        
        return np.mean(self.metric_history[metric_name])
    
    def analyze_posture(self, landmarks, image_shape) -> Optional[PostureMetrics]:
        """
        Analyze posture from MediaPipe landmarks
        
        Key landmarks used:
        - 0: Nose
        - 11: Left shoulder
        - 12: Right shoulder
        - 23: Left hip
        - 24: Right hip
        """
        if not landmarks:
            return None
        
        h, w = image_shape[:2]
        
        # Extract key landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        
        # Convert to pixel coordinates
        nose_coords = (nose.x * w, nose.y * h)
        left_shoulder_coords = (left_shoulder.x * w, left_shoulder.y * h)
        right_shoulder_coords = (right_shoulder.x * w, right_shoulder.y * h)
        left_ear_coords = (left_ear.x * w, left_ear.y * h)
        right_ear_coords = (right_ear.x * w, right_ear.y * h)
        
        # Calculate midpoints
        shoulder_midpoint = (
            (left_shoulder_coords[0] + right_shoulder_coords[0]) / 2,
            (left_shoulder_coords[1] + right_shoulder_coords[1]) / 2
        )
        ear_midpoint = (
            (left_ear_coords[0] + right_ear_coords[0]) / 2,
            (left_ear_coords[1] + right_ear_coords[1]) / 2
        )
        
        # 1. Shoulder angle (should be level)
        shoulder_angle_raw = self.calculate_slope_angle(left_shoulder_coords, right_shoulder_coords)
        shoulder_angle = self.smooth_metric('shoulder_angle', shoulder_angle_raw)
        
        # 2. Head tilt (using ears)
        head_tilt_raw = self.calculate_slope_angle(left_ear_coords, right_ear_coords)
        head_tilt = self.smooth_metric('head_tilt', head_tilt_raw)
        
        # 3. Forward lean (head position relative to shoulders)
        # Calculate vertical distance between nose and shoulder midpoint
        vertical_distance = abs(nose_coords[1] - shoulder_midpoint[1])
        horizontal_offset = abs(nose_coords[0] - shoulder_midpoint[0])
        
        # Ratio of horizontal offset to vertical distance
        forward_lean_raw = horizontal_offset / max(vertical_distance, 1)
        forward_lean = self.smooth_metric('forward_lean', forward_lean_raw)
        
        # Detect issues
        issues = []
        is_slouching = False
        is_tilted = False
        is_leaning = False
        
        if shoulder_angle > self.SHOULDER_ANGLE_MAX or shoulder_angle < self.SHOULDER_ANGLE_MIN:
            is_slouching = True
            issues.append(f"Shoulders not level (angle: {shoulder_angle:.1f}°)")
        
        if head_tilt > self.HEAD_TILT_MAX or head_tilt < self.HEAD_TILT_MIN:
            is_tilted = True
            issues.append(f"Head tilted (angle: {head_tilt:.1f}°)")
        
        if forward_lean > self.FORWARD_LEAN_THRESHOLD:
            is_leaning = True
            issues.append(f"Leaning forward (lean ratio: {forward_lean:.2f})")
        
        return PostureMetrics(
            shoulder_angle=shoulder_angle,
            head_tilt=head_tilt,
            forward_lean=forward_lean,
            is_slouching=is_slouching,
            is_tilted=is_tilted,
            is_leaning=is_leaning,
            timestamp=time.time(),
            issues=issues
        )
    
    def draw_posture_info(self, image, metrics: PostureMetrics):
        """Draw posture information on the image"""
        if not metrics:
            return image
        
        # Set up text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = 30
        
        # Status color (green if good, red if issues)
        status_color = (0, 255, 0) if not metrics.issues else (0, 0, 255)
        
        # Draw background rectangle for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Draw status
        status = "Good Posture" if not metrics.issues else "Posture Issues Detected"
        cv2.putText(image, status, (20, y_offset), font, font_scale, status_color, thickness)
        
        # Draw metrics
        y_offset += 30
        cv2.putText(image, f"Shoulder Angle: {metrics.shoulder_angle:.1f}°", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(image, f"Head Tilt: {metrics.head_tilt:.1f}°", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(image, f"Forward Lean: {metrics.forward_lean:.2f}", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        
        # Draw issues
        if metrics.issues:
            y_offset += 30
            for issue in metrics.issues[:2]:  # Limit to 2 issues displayed
                cv2.putText(image, f"⚠ {issue}", 
                           (20, y_offset), font, 0.4, (0, 165, 255), 1)
                y_offset += 20
        
        return image
    
    def process_frame(self, frame) -> tuple:
        """
        Process a single frame and return annotated frame with metrics
        Returns: (annotated_frame, posture_metrics)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Analyze posture
            metrics = self.analyze_posture(results.pose_landmarks.landmark, frame.shape)
            
            # Draw posture information
            frame = self.draw_posture_info(frame, metrics)
            
            return frame, metrics
        
        return frame, None
    
    def release(self):
        """Clean up resources"""
        self.pose.close()


def main():
    """Main function to run the posture detection system"""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (optional, adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize posture detector
    detector = PostureDetector()
    
    print("Starting posture detection...")
    print("Press 'q' to quit")
    print("Press 's' to save current metrics")
    
    # For tracking metrics over time
    session_metrics = []
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            annotated_frame, metrics = detector.process_frame(frame)
            
            # Store metrics
            if metrics:
                session_metrics.append(metrics)
            
            # Display the frame
            cv2.imshow('Interview Posture Detector', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and metrics:
                print(f"\n--- Current Metrics ---")
                print(f"Shoulder Angle: {metrics.shoulder_angle:.2f}°")
                print(f"Head Tilt: {metrics.head_tilt:.2f}°")
                print(f"Forward Lean: {metrics.forward_lean:.2f}")
                print(f"Issues: {', '.join(metrics.issues) if metrics.issues else 'None'}")
                print("----------------------\n")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        
        # Print session summary
        if session_metrics:
            print("\n=== Session Summary ===")
            print(f"Total frames analyzed: {len(session_metrics)}")
            
            issues_detected = sum(1 for m in session_metrics if m.issues)
            print(f"Frames with issues: {issues_detected} ({issues_detected/len(session_metrics)*100:.1f}%)")
            
            avg_shoulder = np.mean([m.shoulder_angle for m in session_metrics])
            avg_tilt = np.mean([m.head_tilt for m in session_metrics])
            avg_lean = np.mean([m.forward_lean for m in session_metrics])
            
            print(f"\nAverage Metrics:")
            print(f"  Shoulder Angle: {avg_shoulder:.2f}°")
            print(f"  Head Tilt: {avg_tilt:.2f}°")
            print(f"  Forward Lean: {avg_lean:.2f}")
            print("======================\n")


if __name__ == "__main__":
    main()