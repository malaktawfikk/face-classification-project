"""
Real-time Webcam Face Recognition System
Performs live face detection and classification
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import sys
from pathlib import Path
import pickle
from collections import deque
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing.face_detection import FaceDetector


class WebcamFaceRecognition:
    """Real-time face recognition from webcam"""
    
    def __init__(self, model_path: str, encoder_path: str,
                 camera_id: int = 0, confidence_threshold: float = 0.7):
        """
        Initialize webcam recognition system
        
        Args:
            model_path: Path to trained model
            encoder_path: Path to label encoder
            camera_id: Camera device ID
            confidence_threshold: Minimum confidence for recognition
        """
        print("Initializing Webcam Face Recognition System...")
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.input_size = self.model.input_shape[1:3]
        
        # Load label encoder
        print(f"Loading label encoder from {encoder_path}...")
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize face detector
        print("Initializing face detector...")
        self.face_detector = FaceDetector(min_confidence=0.95)
        
        # Configuration
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        
        # Smoothing predictions with moving average
        self.prediction_history = deque(maxlen=5)
        
        # FPS calculation
        self.fps_history = deque(maxlen=30)
        
        print("✅ Initialization complete!")
    
    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face for model
        
        Args:
            face: Face image (BGR)
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Resize
        face_resized = cv2.resize(face_rgb, self.input_size)
        
        # Normalize (ImageNet)
        face_processed = face_resized.astype('float32')
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        face_processed = (face_processed - mean) / std
        
        return face_processed
    
    def predict_identity(self, face: np.ndarray) -> tuple:
        """
        Predict face identity
        
        Args:
            face: Preprocessed face image
            
        Returns:
            (name, confidence)
        """
        # Add batch dimension
        face_batch = np.expand_dims(face, axis=0)
        
        # Predict
        predictions = self.model.predict(face_batch, verbose=0)[0]
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        confidence = predictions[top_idx]
        name = self.label_encoder.inverse_transform([top_idx])[0]
        
        return name, confidence
    
    def smooth_predictions(self, name: str, confidence: float) -> tuple:
        """
        Smooth predictions using moving average
        
        Args:
            name: Predicted name
            confidence: Prediction confidence
            
        Returns:
            (smoothed_name, smoothed_confidence)
        """
        self.prediction_history.append((name, confidence))
        
        # Count occurrences
        name_counts = {}
        confidence_sum = {}
        
        for pred_name, pred_conf in self.prediction_history:
            name_counts[pred_name] = name_counts.get(pred_name, 0) + 1
            confidence_sum[pred_name] = confidence_sum.get(pred_name, 0) + pred_conf
        
        # Get most common name
        most_common = max(name_counts, key=name_counts.get)
        avg_confidence = confidence_sum[most_common] / name_counts[most_common]
        
        return most_common, avg_confidence
    
    def draw_results(self, frame: np.ndarray, detection: dict,
                    name: str, confidence: float, fps: float) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Video frame
            detection: Face detection result
            name: Predicted name
            confidence: Prediction confidence
            fps: Current FPS
            
        Returns:
            Annotated frame
        """
        # Get bounding box
        x, y, width, height = detection['box']
        
        # Choose color based on confidence
        if confidence >= self.confidence_threshold:
            color = (0, 255, 0)  # Green
            label = f"{name} ({confidence:.2%})"
        else:
            color = (0, 165, 255)  # Orange
            label = f"Unknown ({confidence:.2%})"
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_length = int(width * confidence)
        cv2.rectangle(frame, (x, y + height + 5),
                     (x + bar_length, y + height + 15), color, -1)
        cv2.rectangle(frame, (x, y + height + 5),
                     (x + width, y + height + 15), color, 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Run real-time face recognition"""
        print(f"\n🎥 Starting webcam (Camera {self.camera_id})...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Open webcam
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        detection_interval = 5  # Detect every N frames
        
        current_name = "Unknown"
        current_confidence = 0.0
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to read frame")
                break
            
            # Detect and recognize face periodically
            if frame_count % detection_interval == 0:
                # Detect face
                detection = self.face_detector.detect_face(frame)
                
                if detection is not None:
                    # Align face
                    aligned_face = self.face_detector.align_face(
                        frame, detection, self.input_size
                    )
                    
                    if aligned_face is not None:
                        # Preprocess
                        preprocessed = self.preprocess_face(aligned_face)
                        
                        # Predict
                        name, confidence = self.predict_identity(preprocessed)
                        
                        # Smooth predictions
                        current_name, current_confidence = self.smooth_predictions(
                            name, confidence
                        )
                    
                    # Draw results
                    fps = 1.0 / (time.time() - start_time) if len(self.fps_history) > 0 else 0
                    self.fps_history.append(fps)
                    avg_fps = np.mean(self.fps_history)
                    
                    frame = self.draw_results(
                        frame, detection, current_name, current_confidence, avg_fps
                    )
            
            # Display frame
            cv2.imshow('Webcam Face Recognition', frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam recognition stopped")


def main():
    parser = argparse.ArgumentParser(description='Real-time Webcam Face Recognition')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--encoder', type=str, required=True,
                       help='Path to label encoder')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold (default: 0.7)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    if not Path(args.encoder).exists():
        print(f"❌ Error: Encoder file not found: {args.encoder}")
        return
    
    # Initialize and run
    recognizer = WebcamFaceRecognition(
        model_path=args.model,
        encoder_path=args.encoder,
        camera_id=args.camera,
        confidence_threshold=args.threshold
    )
    
    recognizer.run()


if __name__ == "__main__":
    main()
