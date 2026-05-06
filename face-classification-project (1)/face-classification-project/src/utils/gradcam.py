"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Visualization
Model interpretability through heatmap generation
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of conv layer to visualize (auto-detect if None)
        """
        self.model = model
        
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
    
    def _find_last_conv_layer(self) -> str:
        """Auto-detect last convolutional layer"""
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        raise ValueError("No convolutional layer found in model")
    
    def _build_grad_model(self) -> keras.Model:
        """Build gradient model"""
        # Get the conv layer output and model predictions
        conv_layer = self.model.get_layer(self.layer_name)
        
        grad_model = keras.Model(
            inputs=[self.model.inputs],
            outputs=[conv_layer.output, self.model.output]
        )
        
        return grad_model
    
    def compute_heatmap(self, image: np.ndarray, 
                       class_idx: Optional[int] = None,
                       eps: float = 1e-8) -> np.ndarray:
        """
        Compute Grad-CAM heatmap
        
        Args:
            image: Input image (preprocessed)
            class_idx: Target class index (None = predicted class)
            eps: Small value to avoid division by zero
            
        Returns:
            Heatmap as numpy array
        """
        # Expand dimensions if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get conv output and predictions
            conv_outputs, predictions = self.grad_model(image)
            
            # If class_idx not specified, use predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel by corresponding gradient weight
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all channels
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + eps)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap: np.ndarray, image: np.ndarray,
                       alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            image: Original image (RGB, 0-255)
            alpha: Overlay transparency (0-1)
            colormap: OpenCV colormap
            
        Returns:
            Overlaid image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        else:
            image = np.uint8(image)
        
        # Overlay heatmap
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid
    
    def visualize(self, image: np.ndarray, preprocessed_image: np.ndarray,
                 class_idx: Optional[int] = None, class_name: str = "",
                 save_path: Optional[str] = None, show: bool = True):
        """
        Complete visualization: compute and display Grad-CAM
        
        Args:
            image: Original image (for display)
            preprocessed_image: Preprocessed image (for model)
            class_idx: Target class index
            class_name: Class name for title
            save_path: Path to save figure
            show: Display figure
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(preprocessed_image, class_idx)
        
        # Create overlay
        overlaid = self.overlay_heatmap(heatmap, image)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlaid)
        title = 'Grad-CAM Overlay'
        if class_name:
            title += f'\nClass: {class_name}'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, overlaid


def visualize_multiple_classes(gradcam: GradCAM, image: np.ndarray,
                               preprocessed_image: np.ndarray,
                               top_k: int = 5, label_names: list = None,
                               save_path: Optional[str] = None):
    """
    Visualize Grad-CAM for top-k predicted classes
    
    Args:
        gradcam: GradCAM instance
        image: Original image
        preprocessed_image: Preprocessed image
        top_k: Number of top predictions to visualize
        label_names: List of class names
        save_path: Path to save figure
    """
    # Get predictions
    if len(preprocessed_image.shape) == 3:
        preprocessed_image_batch = np.expand_dims(preprocessed_image, axis=0)
    else:
        preprocessed_image_batch = preprocessed_image
    
    predictions = gradcam.model.predict(preprocessed_image_batch, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    
    # Create figure
    fig, axes = plt.subplots(2, top_k, figsize=(4 * top_k, 8))
    
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        # Compute heatmap
        heatmap = gradcam.compute_heatmap(preprocessed_image, class_idx=idx)
        overlaid = gradcam.overlay_heatmap(heatmap, image)
        
        # Display heatmap
        axes[0, i].imshow(heatmap, cmap='jet')
        class_label = label_names[idx] if label_names else f"Class {idx}"
        axes[0, i].set_title(f'{class_label}\n({prob:.2%})')
        axes[0, i].axis('off')
        
        # Display overlay
        axes[1, i].imshow(overlaid)
        axes[1, i].axis('off')
    
    plt.suptitle(f'Top-{top_k} Predictions with Grad-CAM', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--layer', type=str, default=None,
                       help='Conv layer name (auto-detect if not specified)')
    parser.add_argument('--output', type=str, default='gradcam_output.png',
                       help='Output path')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)
    
    # Load and preprocess image
    print(f"Loading image from {args.image}...")
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and preprocess
    input_size = model.input_shape[1:3]
    preprocessed = cv2.resize(image, input_size)
    preprocessed = preprocessed.astype('float32')
    
    # ImageNet normalization
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    preprocessed = (preprocessed - mean) / std
    
    # Create Grad-CAM
    print("Generating Grad-CAM visualization...")
    gradcam = GradCAM(model, layer_name=args.layer)
    
    # Visualize
    gradcam.visualize(
        image=image,
        preprocessed_image=preprocessed,
        save_path=args.output,
        show=True
    )
    
    print("Done!")
