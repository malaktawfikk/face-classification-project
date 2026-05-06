# Quick Start Guide
## Face Classification Project

This guide will help you get started with the Face Classification project in minutes.

## 📋 Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git
- (Optional) CUDA-capable GPU for faster training

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/KarimHabib100/face-classification-project.git
cd face-classification-project
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Run Setup Script

```bash
python setup.py
```

This will:
- Check Python version
- Create necessary directories
- Install all dependencies
- Check GPU availability
- Create configuration files
- Optionally download the LFW dataset

### Step 4: Verify Installation

```bash
# Test imports
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## 📊 Dataset Preparation

### Option 1: LFW Dataset (Recommended for beginners)

```bash
# Download dataset
python src/utils/download_data.py --dataset lfw

# Process faces (detect and align)
python src/preprocessing/face_detection.py \
    --input data/raw/lfw \
    --output data/processed
```

### Option 2: Create Sample Dataset (Quick testing)

```bash
# Download and create small sample
python src/utils/download_data.py --dataset lfw --create-sample

# Process sample
python src/preprocessing/face_detection.py \
    --input data/sample \
    --output data/processed_sample
```

## 🎓 Training Models

### Train ResNet50 (Recommended)

```bash
python src/train.py \
    --model resnet50 \
    --epochs 50 \
    --batch-size 32
```

### Train InceptionV3

```bash
python src/train.py \
    --model inceptionv3 \
    --epochs 50 \
    --batch-size 32
```

### Train EfficientNetB0 (Fastest)

```bash
python src/train.py \
    --model efficientnet \
    --epochs 50 \
    --batch-size 32
```

### Quick Training (Testing)

```bash
# Train on sample dataset for 10 epochs
python src/train.py \
    --model efficientnet \
    --epochs 10 \
    --batch-size 16
```

## 📈 Model Evaluation

```bash
python src/evaluate.py \
    --model data/models/resnet50/resnet50_best.keras \
    --encoder data/models/resnet50/label_encoder.pkl \
    --test-data data/processed
```

This generates:
- Confusion matrix
- ROC curves
- Top-K accuracy plots
- Error analysis report

## 🎯 Using Trained Models

### 1. Predict Single Image

```bash
python src/predict.py \
    --model data/models/resnet50/resnet50_best.keras \
    --encoder data/models/resnet50/label_encoder.pkl \
    --image path/to/image.jpg
```

### 2. Grad-CAM Visualization

```bash
python src/utils/gradcam.py \
    --model data/models/resnet50/resnet50_best.keras \
    --image path/to/image.jpg \
    --output gradcam_result.png
```

### 3. Streamlit Web Interface

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to: `http://localhost:8501`

### 4. Real-time Webcam Recognition

```bash
python app/webcam_recognition.py \
    --model data/models/resnet50/resnet50_best.keras \
    --encoder data/models/resnet50/label_encoder.pkl \
    --camera 0 \
    --threshold 0.7
```

Controls:
- Press `q` to quit
- Press `s` to save screenshot

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Training parameters
training:
  epochs: 50
  batch_size: 32
  initial_learning_rate: 0.001

# Model architecture
models:
  resnet50:
    freeze_layers: 143
    dropout_rate: 0.5

# Preprocessing
preprocessing:
  target_size: [224, 224]
  augmentation:
    enabled: true
```

## 📊 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

Open browser to: `http://localhost:6006`

### Training Logs

View training progress:
```bash
# View CSV log
cat logs/resnet50_YYYYMMDD_HHMMSS/training_log.csv

# Monitor in real-time
tail -f logs/training.log
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, check CUDA installation
nvidia-smi
```

### Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 16  # or 8
```

### Slow Training

- Use smaller model (EfficientNetB0)
- Reduce image size
- Use GPU if available
- Reduce number of classes

### Face Detection Fails

- Adjust confidence threshold:
```bash
python src/preprocessing/face_detection.py \
    --confidence 0.90  # Lower threshold
```

### Module Import Errors

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 📝 Common Workflows

### Experiment with Different Models

```bash
# Train all three models
for model in resnet50 inceptionv3 efficientnet; do
    python src/train.py --model $model --epochs 50
done

# Compare results
python src/evaluate.py --model data/models/resnet50/resnet50_best.keras ...
python src/evaluate.py --model data/models/inceptionv3/inceptionv3_best.keras ...
python src/evaluate.py --model data/models/efficientnet/efficientnet_best.keras ...
```

### Fine-tuning

```bash
# Load pre-trained model and train more epochs
python src/train.py \
    --model resnet50 \
    --epochs 20 \
    --load-weights data/models/resnet50/resnet50_best.keras
```

### Batch Prediction

```bash
# Predict all images in directory
for img in data/test_images/*.jpg; do
    python src/predict.py \
        --model data/models/resnet50/resnet50_best.keras \
        --encoder data/models/resnet50/label_encoder.pkl \
        --image "$img"
done
```

## 🎯 Project Structure Overview

```
face-classification-project/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed faces
│   └── models/          # Trained models
├── src/
│   ├── models/          # Model architectures
│   ├── preprocessing/   # Data preprocessing
│   ├── utils/           # Utility functions
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
├── app/
│   ├── streamlit_app.py      # Web interface
│   └── webcam_recognition.py # Real-time recognition
├── config/
│   └── config.yaml      # Configuration
└── requirements.txt     # Dependencies
```

## 📚 Next Steps

1. **Explore the notebooks** in `notebooks/` for detailed analysis
2. **Read the full README** for comprehensive documentation
3. **Check the examples** in the repository
4. **Customize the models** in `src/models/`
5. **Contribute** improvements via pull requests

## 💡 Tips

- Start with a small sample dataset to test your setup
- Use EfficientNetB0 for quick experiments
- Monitor training with TensorBoard
- Save models frequently during training
- Document your experiments in notebooks

## 🆘 Getting Help

- Check the [README.md](README.md) for detailed documentation
- Open an issue on GitHub for bugs
- Review closed issues for solutions
- Check TensorFlow/Keras documentation

## 🎓 Learning Resources

- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Face Recognition Papers](https://paperswithcode.com/task/face-recognition)
- [Computer Vision Course](https://www.coursera.org/specializations/deep-learning)

---

**Ready to start?** Run `python setup.py` and follow the prompts!
