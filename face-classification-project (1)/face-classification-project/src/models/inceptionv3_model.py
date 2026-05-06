"""
InceptionV3 Model for Face Classification
Transfer learning implementation with fine-tuning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from typing import Tuple, Optional
import yaml


class InceptionV3FaceClassifier:
    """InceptionV3-based face classification model"""
    
    def __init__(self, num_classes: int, config: dict = None):
        """
        Initialize InceptionV3 model
        
        Args:
            num_classes: Number of identities to classify
            config: Configuration dictionary
        """
        self.num_classes = num_classes
        self.config = config or self._default_config()
        self.model = None
        
    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            'input_size': [299, 299, 3],
            'pretrained_weights': 'imagenet',
            'freeze_layers': 249,
            'dropout_rate': 0.5,
            'dense_units': 512
        }
    
    def build_model(self) -> Model:
        """
        Build InceptionV3 model with transfer learning
        
        Returns:
            Compiled Keras model
        """
        input_shape = tuple(self.config['input_size'])
        
        # Load pre-trained InceptionV3
        base_model = InceptionV3(
            weights=self.config['pretrained_weights'],
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model layers
        freeze_layers = self.config['freeze_layers']
        for layer in base_model.layers[:freeze_layers]:
            layer.trainable = False
        
        # Build top layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        
        # First dense layer
        x = layers.Dense(
            self.config['dense_units'],
            activation='relu',
            name='dense_1',
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(self.config['dropout_rate'], name='dropout_1')(x)
        
        # Second dense layer
        x = layers.Dense(
            self.config['dense_units'] // 2,
            activation='relu',
            name='dense_2',
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(self.config['dropout_rate'], name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create model
        self.model = Model(
            inputs=base_model.input,
            outputs=outputs,
            name='InceptionV3_FaceClassifier'
        )
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile model with optimizer and loss
        
        Args:
            learning_rate: Initial learning rate
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
    
    def unfreeze_layers(self, num_layers: Optional[int] = None):
        """
        Unfreeze layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        if self.model is None:
            raise ValueError("Model not built.")
        
        if num_layers is None:
            # Unfreeze all layers
            for layer in self.model.layers:
                layer.trainable = True
        else:
            # Unfreeze last N layers
            for layer in self.model.layers[-num_layers:]:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.summary()
    
    def get_gradcam_layer(self) -> str:
        """
        Get the name of the last convolutional layer for Grad-CAM
        
        Returns:
            Layer name
        """
        # Find last conv layer
        for layer in reversed(self.model.layers):
            if isinstance(layer, layers.Conv2D):
                return layer.name
        
        # Fallback to mixed10 (last inception module)
        return 'mixed10'
    
    @staticmethod
    def load_from_config(config_path: str, num_classes: int) -> 'InceptionV3FaceClassifier':
        """
        Load model configuration from YAML file
        
        Args:
            config_path: Path to config file
            num_classes: Number of classes
            
        Returns:
            InceptionV3FaceClassifier instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config['models']['inceptionv3']
        return InceptionV3FaceClassifier(num_classes, model_config)


def create_inceptionv3_model(num_classes: int,
                             input_shape: Tuple[int, int, int] = (299, 299, 3),
                             learning_rate: float = 0.001) -> Model:
    """
    Convenience function to create and compile InceptionV3 model
    
    Args:
        num_classes: Number of classes
        input_shape: Input image shape
        learning_rate: Learning rate
        
    Returns:
        Compiled Keras model
    """
    classifier = InceptionV3FaceClassifier(num_classes)
    classifier.config['input_size'] = list(input_shape)
    
    model = classifier.build_model()
    classifier.compile_model(learning_rate)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating InceptionV3 Face Classifier...")
    
    # Create model
    classifier = InceptionV3FaceClassifier(num_classes=100)
    model = classifier.build_model()
    classifier.compile_model(learning_rate=0.001)
    
    # Print summary
    print("\nModel Summary:")
    classifier.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"Grad-CAM layer: {classifier.get_gradcam_layer()}")
