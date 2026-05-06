"""
Streamlit Web Application for Face Classification
Interactive GUI for model inference and visualization
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.gradcam import GradCAM
from preprocessing.face_detection import FaceDetector


# Page configuration
st.set_page_config(
    page_title="Face Classification System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load trained model"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_label_encoder(encoder_path):
    """Load label encoder"""
    try:
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        return encoder
    except Exception as e:
        st.error(f"Error loading label encoder: {e}")
        return None


@st.cache_resource
def initialize_face_detector():
    """Initialize face detector"""
    return FaceDetector(min_confidence=0.95)


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model"""
    # Resize
    image_resized = cv2.resize(image, target_size)
    
    # Convert to float and normalize (ImageNet)
    image_processed = image_resized.astype('float32')
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    image_processed = (image_processed - mean) / std
    
    return image_processed


def predict_face(model, image, label_encoder, top_k=5):
    """Predict face identity"""
    # Add batch dimension
    image_batch = np.expand_dims(image, axis=0)
    
    # Predict
    predictions = model.predict(image_batch, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_labels = label_encoder.inverse_transform(top_indices)
    
    results = []
    for label, prob in zip(top_labels, top_probs):
        results.append({
            'name': label,
            'confidence': float(prob)
        })
    
    return results


def main():
    # Header
    st.markdown('<h1 class="main-header">👤 Face Classification System</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "ResNet50": "data/models/resnet50/resnet50_best.keras",
            "InceptionV3": "data/models/inceptionv3/inceptionv3_best.keras",
            "EfficientNetB0": "data/models/efficientnet/efficientnet_best.keras"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys())
        )
        
        model_path = model_options[selected_model]
        
        # Encoder path
        encoder_path = f"data/models/{selected_model.lower()}/label_encoder.pkl"
        
        st.markdown("---")
        
        # Options
        st.subheader("Options")
        detect_face = st.checkbox("Auto-detect face", value=True)
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        top_k = st.slider("Top-K Predictions", min_value=1, max_value=10, value=5)
        
        st.markdown("---")
        st.info("Upload an image to classify the person's identity")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', 
                   unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a face image for classification"
        )
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process button
            if st.button("🔍 Classify Face", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Load model and encoder
                    model = load_model(model_path)
                    label_encoder = load_label_encoder(encoder_path)
                    
                    if model is None or label_encoder is None:
                        st.error("Failed to load model or label encoder")
                        return
                    
                    # Get input size
                    input_size = model.input_shape[1:3]
                    
                    # Detect face if enabled
                    if detect_face:
                        face_detector = initialize_face_detector()
                        aligned_face = face_detector.detect_and_align(
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                            target_size=input_size
                        )
                        
                        if aligned_face is not None:
                            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                            processed_image = preprocess_image(aligned_face, input_size)
                            display_image = aligned_face
                            st.success("✅ Face detected and aligned")
                        else:
                            st.warning("⚠️ No face detected, using full image")
                            processed_image = preprocess_image(image, input_size)
                            display_image = cv2.resize(image, input_size)
                    else:
                        processed_image = preprocess_image(image, input_size)
                        display_image = cv2.resize(image, input_size)
                    
                    # Predict
                    results = predict_face(model, processed_image, label_encoder, top_k)
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['display_image'] = display_image
                    st.session_state['processed_image'] = processed_image
                    st.session_state['model'] = model
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Results</h2>', 
                   unsafe_allow_html=True)
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            display_image = st.session_state['display_image']
            
            # Display processed image
            st.image(display_image, caption="Processed Face", use_container_width=True)
            
            # Top prediction
            top_result = results[0]
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1E88E5;">🎯 Top Prediction</h3>
                <h2>{top_result['name']}</h2>
                <h3>Confidence: {top_result['confidence']:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # All predictions
            st.markdown("### 📋 All Predictions")
            for i, result in enumerate(results, 1):
                confidence = result['confidence']
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{i}. {result['name']}</strong>
                    <div style="display: flex; align-items: center;">
                        <div style="flex-grow: 1; background-color: #ddd; border-radius: 10px; margin: 0.5rem 0;">
                            <div style="width: {confidence*100}%; background-color: #1E88E5; 
                                        height: 20px; border-radius: 10px;"></div>
                        </div>
                        <span style="margin-left: 1rem;"><strong>{confidence:.2%}</strong></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Grad-CAM visualization
            if show_gradcam:
                st.markdown("---")
                st.markdown("### 🔥 Grad-CAM Visualization")
                
                with st.spinner("Generating Grad-CAM..."):
                    model = st.session_state['model']
                    processed_image = st.session_state['processed_image']
                    
                    gradcam = GradCAM(model)
                    heatmap = gradcam.compute_heatmap(processed_image)
                    overlaid = gradcam.overlay_heatmap(heatmap, display_image)
                    
                    col_grad1, col_grad2 = st.columns(2)
                    with col_grad1:
                        st.image(heatmap, caption="Heatmap", use_container_width=True, 
                                clamp=True, channels="RGB")
                    with col_grad2:
                        st.image(overlaid, caption="Overlay", use_container_width=True)
        else:
            st.info("👈 Upload an image and click 'Classify Face' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Face Classification System | Built with Streamlit & TensorFlow</p>
        <p>Model: {selected_model} | Powered by Deep Learning</p>
    </div>
    """.format(selected_model=selected_model), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
