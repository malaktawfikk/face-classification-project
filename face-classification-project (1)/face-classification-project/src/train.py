"""
Main Training Script for Face Classification
Supports ResNet50, InceptionV3, and EfficientNetB0
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.resnet50_model import ResNet50FaceClassifier
from models.inceptionv3_model import InceptionV3FaceClassifier
from models.efficientnet_model import EfficientNetFaceClassifier
from utils.data_loader import FaceDataLoader


class FaceClassificationTrainer:
    """Trainer for face classification models"""
    
    def __init__(self, config_path: str, model_name: str):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
            model_name: Model architecture (resnet50, inceptionv3, efficientnet)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = model_name.lower()
        self.model = None
        self.history = None
        
        # Set random seeds
        self._set_seeds()
        
        # Setup paths
        self._setup_paths()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config.get('random_seed', 42)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def _setup_paths(self):
        """Create necessary directories"""
        paths = self.config['paths']
        for path_key, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
        
        # Create model-specific directory
        self.model_dir = Path(paths['models_dir']) / self.model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(paths['logs_dir']) / f"{self.model_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and prepare dataset"""
        print("=" * 70)
        print("Loading Dataset")
        print("=" * 70)
        
        dataset_config = self.config['dataset']
        
        # Get target size based on model
        model_config = self.config['models'][self.model_name]
        target_size = tuple(model_config['input_size'][:2])
        
        # Load data
        loader = FaceDataLoader(
            data_dir=dataset_config['processed_dir'],
            target_size=target_size,
            min_images_per_person=dataset_config['min_faces_per_person']
        )
        
        images, labels, label_names = loader.load_dataset()
        
        # Save label encoder
        encoder_path = self.model_dir / 'label_encoder.pkl'
        loader.save_label_encoder(str(encoder_path))
        
        # Split dataset
        data = loader.split_dataset(
            train_ratio=dataset_config['train_split'],
            val_ratio=dataset_config['val_split'],
            test_ratio=dataset_config['test_split']
        )
        
        # Preprocess data
        preprocessing_config = self.config['preprocessing']
        data['X_train'] = loader.preprocess_data(
            data['X_train'],
            normalization=preprocessing_config['normalization']
        )
        data['X_val'] = loader.preprocess_data(
            data['X_val'],
            normalization=preprocessing_config['normalization']
        )
        data['X_test'] = loader.preprocess_data(
            data['X_test'],
            normalization=preprocessing_config['normalization']
        )
        
        # Create data generators
        training_config = self.config['training']
        train_gen, val_gen = loader.create_data_generators(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            batch_size=training_config['batch_size'],
            augmentation=preprocessing_config['augmentation']['enabled']
        )
        
        self.loader = loader
        self.data = data
        self.train_generator = train_gen
        self.val_generator = val_gen
        self.num_classes = len(label_names)
        
        print(f"\nNumber of classes: {self.num_classes}")
        print(f"Sample classes: {label_names[:10]}")
    
    def build_model(self):
        """Build model architecture"""
        print("\n" + "=" * 70)
        print(f"Building {self.model_name.upper()} Model")
        print("=" * 70)
        
        model_config = self.config['models'][self.model_name]
        
        if self.model_name == 'resnet50':
            classifier = ResNet50FaceClassifier(self.num_classes, model_config)
        elif self.model_name == 'inceptionv3':
            classifier = InceptionV3FaceClassifier(self.num_classes, model_config)
        elif self.model_name == 'efficientnet':
            classifier = EfficientNetFaceClassifier(self.num_classes, model_config)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.model = classifier.build_model()
        
        # Compile model
        training_config = self.config['training']
        classifier.compile_model(training_config['initial_learning_rate'])
        
        self.classifier = classifier
        
        # Print summary
        print("\nModel Architecture:")
        self.model.summary()
        
        print(f"\nTotal parameters: {self.model.count_params():,}")
        trainable = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable:,}")
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = []
        training_config = self.config['training']
        
        # Model checkpoint
        if training_config['callbacks']['model_checkpoint']:
            checkpoint_path = self.model_dir / f'{self.model_name}_best.keras'
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
            )
        
        # Early stopping
        if training_config['early_stopping']['enabled']:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=training_config['early_stopping']['monitor'],
                    patience=training_config['early_stopping']['patience'],
                    restore_best_weights=training_config['early_stopping']['restore_best_weights'],
                    verbose=1
                )
            )
        
        # Learning rate schedule
        lr_config = training_config['lr_schedule']
        if lr_config['type'] == 'reduce_on_plateau':
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=lr_config['factor'],
                    patience=lr_config['patience'],
                    min_lr=lr_config['min_lr'],
                    verbose=1
                )
            )
        
        # TensorBoard
        if training_config['callbacks']['tensorboard']:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=str(self.run_dir),
                    histogram_freq=1,
                    write_graph=True
                )
            )
        
        # CSV Logger
        if training_config['callbacks']['csv_logger']:
            callbacks.append(
                keras.callbacks.CSVLogger(
                    str(self.run_dir / 'training_log.csv')
                )
            )
        
        return callbacks
    
    def train(self):
        """Train the model"""
        print("\n" + "=" * 70)
        print("Training Model")
        print("=" * 70)
        
        training_config = self.config['training']
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Calculate steps per epoch
        steps_per_epoch = len(self.data['X_train']) // training_config['batch_size']
        validation_steps = len(self.data['X_val']) // training_config['batch_size']
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=training_config['epochs'],
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = self.model_dir / f'{self.model_name}_final.keras'
        self.model.save(str(final_model_path))
        print(f"\nFinal model saved to: {final_model_path}")
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "=" * 70)
        print("Evaluating Model")
        print("=" * 70)
        
        # Prepare test data
        from tensorflow.keras.utils import to_categorical
        X_test = self.data['X_test']
        y_test_cat = to_categorical(self.data['y_test'], self.num_classes)
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test_cat, verbose=1)
        
        print("\nTest Results:")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"  {metric_name}: {value:.4f}")
        
        # Save results
        results_file = self.model_dir / 'test_results.txt'
        with open(results_file, 'w') as f:
            for metric_name, value in zip(self.model.metrics_names, results):
                f.write(f"{metric_name}: {value:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Train face classification model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'inceptionv3', 'efficientnet'],
                       help='Model architecture')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FaceClassificationTrainer(args.config, args.model)
    
    # Override config if specified
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    
    # Run training pipeline
    trainer.load_data()
    trainer.build_model()
    trainer.train()
    trainer.evaluate()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
