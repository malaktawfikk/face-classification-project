"""
Simple Prediction Script
Predict face identity from single image
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.face_detection import FaceDetector


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image, target_size)
    
    # Normalize (ImageNet)
    image_processed = image_resized.astype('float32')
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    image_processed = (image_processed - mean) / std
    
    return image_processed


def predict_face(model_path, encoder_path, image_path, 
                detect_face=True, top_k=5, show_image=True):
    """
    Predict face identity from image
    
    Args:
        model_path: Path to trained model
        encoder_path: Path to label encoder
        image_path: Path to input image
        detect_face: Auto-detect and align face
        top_k: Number of top predictions to show
        show_image: Display image with results
    """
    print("=" * 70)
    print("Face Classification - Prediction")
    print("=" * 70)
    
    # Load model
    print(f"\n📦 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    input_size = model.input_shape[1:3]
    
    # Load label encoder
    print(f"📦 Loading label encoder from {encoder_path}...")
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load image
    print(f"📷 Loading image from {image_path}...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    
    original_image = image.copy()
    
    # Detect face if requested
    if detect_face:
        print("🔍 Detecting face...")
        face_detector = FaceDetector(min_confidence=0.95)
        
        aligned_face = face_detector.detect_and_align(image, target_size=input_size)
        
        if aligned_face is not None:
            print("✅ Face detected and aligned")
            image = aligned_face
        else:
            print("⚠️  No face detected, using full image")
    
    # Preprocess
    print("🔄 Preprocessing image...")
    processed_image = preprocess_image(image, target_size=input_size)
    
    # Predict
    print("🤖 Making prediction...")
    image_batch = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(image_batch, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_labels = label_encoder.inverse_transform(top_indices)
    
    # Display results
    print("\n" + "=" * 70)
    print(f"Top {top_k} Predictions")
    print("=" * 70)
    
    for i, (label, prob) in enumerate(zip(top_labels, top_probs), 1):
        bar_length = int(50 * prob)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{i}. {label:20s} [{bar}] {prob:.2%}")
    
    # Show image if requested
    if show_image:
        # Create result image
        display_image = cv2.resize(original_image, (400, 400))
        
        # Add prediction text
        text = f"{top_labels[0]} ({top_probs[0]:.1%})"
        cv2.putText(display_image, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Prediction Result', display_image)
        print("\n👁️  Displaying result (press any key to close)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return top_labels[0], top_probs[0]


def main():
    parser = argparse.ArgumentParser(description='Predict face identity from image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--encoder', type=str, required=True,
                       help='Path to label encoder')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--no-detect', action='store_true',
                       help='Skip face detection')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions (default: 5)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display image')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    if not Path(args.encoder).exists():
        print(f"❌ Error: Encoder file not found: {args.encoder}")
        return
    
    if not Path(args.image).exists():
        print(f"❌ Error: Image file not found: {args.image}")
        return
    
    # Run prediction
    predict_face(
        model_path=args.model,
        encoder_path=args.encoder,
        image_path=args.image,
        detect_face=not args.no_detect,
        top_k=args.top_k,
        show_image=not args.no_display
    )


if __name__ == "__main__":
    main()
