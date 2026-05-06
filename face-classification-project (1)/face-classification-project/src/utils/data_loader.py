"""
Data Loader for Face Classification
Handles dataset loading, splitting, and augmentation
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import pickle


class FaceDataLoader:
    """Load and prepare face classification dataset"""
    
    def __init__(self, data_dir: str, target_size: Tuple[int, int] = (224, 224),
                 min_images_per_person: int = 10):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing face images organized by person
            target_size: Target image size (height, width)
            min_images_per_person: Minimum images required per person
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.min_images_per_person = min_images_per_person
        self.label_encoder = LabelEncoder()
        
        self.images = []
        self.labels = []
        self.label_names = []
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load dataset from directory structure
        
        Returns:
            images, labels, label_names
        """
        print("Loading dataset...")
        
        # Get all person directories
        person_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        temp_images = []
        temp_labels = []
        
        for person_dir in person_dirs:
            person_name = person_dir.name
            
            # Get all images for this person
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(person_dir.glob(ext)))
            
            # Skip if not enough images
            if len(image_files) < self.min_images_per_person:
                continue
            
            # Load images
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.target_size)
                    temp_images.append(img)
                    temp_labels.append(person_name)
        
        print(f"Loaded {len(temp_images)} images from {len(set(temp_labels))} people")
        
        # Encode labels
        self.labels = self.label_encoder.fit_transform(temp_labels)
        self.label_names = list(self.label_encoder.classes_)
        self.images = np.array(temp_images)
        
        return self.images, self.labels, self.label_names
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                     test_ratio: float = 0.15, random_state: int = 42) -> Dict:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
            
        Returns:
            Dictionary with split data
        """
        assert train_ratio + val_ratio + test_ratio == 1.0
        
        # First split: train and temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.images, self.labels,
            test_size=(1 - train_ratio),
            random_state=random_state,
            stratify=self.labels
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(X_train)} images")
        print(f"  Validation: {len(X_val)} images")
        print(f"  Test: {len(X_test)} images")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def preprocess_data(self, X: np.ndarray, normalization: str = 'imagenet') -> np.ndarray:
        """
        Preprocess images
        
        Args:
            X: Input images
            normalization: Normalization method ('imagenet', 'standard', 'minmax')
            
        Returns:
            Preprocessed images
        """
        X = X.astype('float32')
        
        if normalization == 'imagenet':
            # ImageNet normalization
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            X = (X - mean) / std
        elif normalization == 'standard':
            # Zero mean, unit variance
            X = (X - np.mean(X)) / np.std(X)
        elif normalization == 'minmax':
            # Scale to [0, 1]
            X = X / 255.0
        
        return X
    
    def create_data_generators(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               batch_size: int = 32,
                               augmentation: bool = True) -> Tuple:
        """
        Create data generators with augmentation
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size
            augmentation: Apply data augmentation
            
        Returns:
            train_generator, val_generator
        """
        if augmentation:
            # Training data augmentation
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Convert labels to categorical
        num_classes = len(self.label_names)
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train_cat,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val_cat,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def save_label_encoder(self, filepath: str):
        """Save label encoder"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_label_encoder(self, filepath: str):
        """Load label encoder"""
        with open(filepath, 'rb') as f:
            self.label_encoder = pickle.load(f)
            self.label_names = list(self.label_encoder.classes_)
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            y: Labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        return dict(zip(classes, weights))


def create_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                     shuffle: bool = True, augment: bool = False) -> tf.data.Dataset:
    """
    Create TensorFlow dataset
    
    Args:
        X: Images
        y: Labels
        batch_size: Batch size
        shuffle: Shuffle data
        augment: Apply augmentation
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    if augment:
        def augment_fn(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            return image, label
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Example usage
    print("Face Data Loader Example\n")
    
    data_dir = "data/processed"
    
    if Path(data_dir).exists():
        loader = FaceDataLoader(data_dir, target_size=(224, 224))
        images, labels, label_names = loader.load_dataset()
        
        print(f"\nDataset statistics:")
        print(f"  Total images: {len(images)}")
        print(f"  Number of people: {len(label_names)}")
        print(f"  Image shape: {images[0].shape}")
        
        # Split dataset
        data = loader.split_dataset()
        
        # Create generators
        X_train_norm = loader.preprocess_data(data['X_train'])
        X_val_norm = loader.preprocess_data(data['X_val'])
        
        train_gen, val_gen = loader.create_data_generators(
            X_train_norm, data['y_train'],
            X_val_norm, data['y_val'],
            batch_size=32
        )
        
        print(f"\nGenerators created successfully")
    else:
        print(f"Data directory {data_dir} not found")
