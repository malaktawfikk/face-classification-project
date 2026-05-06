"""
Face Detection and Alignment Module
Uses MTCNN for robust face detection and alignment
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from typing import Tuple, List, Optional
import os
from pathlib import Path
from tqdm import tqdm


class FaceDetector:
    """Face detection and alignment using MTCNN"""
    
    def __init__(self, min_confidence: float = 0.95):
        """
        Initialize face detector
        
        Args:
            min_confidence: Minimum detection confidence threshold
        """
        self.detector = MTCNN()
        self.min_confidence = min_confidence
    
    def detect_face(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect face in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Detection result with bounding box and landmarks, or None
        """
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_image)
        
        # Return best detection above confidence threshold
        if detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            if best_detection['confidence'] >= self.min_confidence:
                return best_detection
        
        return None
    
    def align_face(self, image: np.ndarray, detection: dict, 
                   target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Align and crop face based on detection
        
        Args:
            image: Input image
            detection: Face detection result
            target_size: Output image size (width, height)
            
        Returns:
            Aligned face image or None
        """
        # Extract bounding box
        x, y, width, height = detection['box']
        x, y = abs(x), abs(y)
        
        # Add margin
        margin = 0.2
        x_margin = int(width * margin)
        y_margin = int(height * margin)
        
        x1 = max(0, x - x_margin)
        y1 = max(0, y - y_margin)
        x2 = min(image.shape[1], x + width + x_margin)
        y2 = min(image.shape[0], y + height + y_margin)
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
        
        return face
    
    def detect_and_align(self, image: np.ndarray, 
                        target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Detect and align face in one step
        
        Args:
            image: Input image
            target_size: Output image size
            
        Returns:
            Aligned face image or None
        """
        detection = self.detect_face(image)
        if detection is None:
            return None
        
        return self.align_face(image, detection, target_size)
    
    def draw_detection(self, image: np.ndarray, detection: dict, 
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw bounding box and landmarks on image
        
        Args:
            image: Input image
            detection: Face detection result
            color: Drawing color (BGR)
            
        Returns:
            Image with drawn detection
        """
        image_copy = image.copy()
        
        # Draw bounding box
        x, y, width, height = detection['box']
        cv2.rectangle(image_copy, (x, y), (x + width, y + height), color, 2)
        
        # Draw confidence
        conf_text = f"{detection['confidence']:.2f}"
        cv2.putText(image_copy, conf_text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw landmarks
        for key, point in detection['keypoints'].items():
            cv2.circle(image_copy, point, 2, (0, 0, 255), 2)
        
        return image_copy


def process_dataset(input_dir: str, output_dir: str, 
                    target_size: Tuple[int, int] = (224, 224),
                    min_confidence: float = 0.95):
    """
    Process entire dataset: detect and align all faces
    
    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save processed images
        target_size: Output image size
        min_confidence: Minimum detection confidence
    """
    detector = FaceDetector(min_confidence)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images to process")
    
    success_count = 0
    fail_count = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            fail_count += 1
            continue
        
        # Detect and align face
        aligned_face = detector.detect_and_align(image, target_size)
        
        if aligned_face is not None:
            # Create output directory structure
            relative_path = img_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save aligned face
            cv2.imwrite(str(output_file), aligned_face)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} images")
    print(f"Failed to process: {fail_count} images")
    print(f"Success rate: {success_count / len(image_files) * 100:.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Detection and Alignment')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with raw images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for processed images')
    parser.add_argument('--size', type=int, default=224,
                       help='Target image size (default: 224)')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Minimum detection confidence (default: 0.95)')
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=(args.size, args.size),
        min_confidence=args.confidence
    )
