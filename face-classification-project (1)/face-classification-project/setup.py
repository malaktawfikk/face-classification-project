"""
Setup Script for Face Classification Project
Automated installation and configuration
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")


def run_command(command, description):
    """Run shell command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"   Output: {e.output}")
        return False


def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher required")
        return False
    
    print("✅ Python version is compatible")
    return True


def setup_directories():
    """Create necessary directories"""
    print_header("Creating Project Directories")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "results/plots",
        "results/reports",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True


def install_requirements():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Upgrade pip
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    
    # Install requirements
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements"
    )


def check_gpu():
    """Check GPU availability"""
    print_header("Checking GPU Support")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            return True
        else:
            print("⚠️  No GPU found. Training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def test_imports():
    """Test if key packages can be imported"""
    print_header("Testing Package Imports")
    
    packages = [
        ("tensorflow", "TensorFlow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("streamlit", "Streamlit"),
        ("mtcnn", "MTCNN")
    ]
    
    all_success = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Failed to import")
            all_success = False
    
    return all_success


def create_sample_config():
    """Create sample .env file"""
    print_header("Creating Configuration Files")
    
    env_content = """# Face Classification Project Configuration

# Paths
DATA_DIR=data
MODELS_DIR=data/models
RESULTS_DIR=results

# Training
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.001

# Webcam
CAMERA_ID=0
CONFIDENCE_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")
    
    return True


def download_test_data():
    """Ask user if they want to download test data"""
    print_header("Dataset Download")
    
    print("Would you like to download the LFW dataset now?")
    print("(You can always do this later by running:")
    print("  python src/utils/download_data.py --dataset lfw)")
    print()
    
    response = input("Download now? (y/n): ").lower().strip()
    
    if response == 'y':
        return run_command(
            f"{sys.executable} src/utils/download_data.py --dataset lfw",
            "Downloading LFW dataset"
        )
    else:
        print("⏭️  Skipping dataset download")
        return True


def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print("🎉 Your Face Classification project is ready!")
    print()
    print("📝 Next Steps:")
    print()
    print("1. Download dataset (if not done):")
    print("   python src/utils/download_data.py --dataset lfw")
    print()
    print("2. Preprocess faces:")
    print("   python src/preprocessing/face_detection.py \\")
    print("      --input data/raw/lfw --output data/processed")
    print()
    print("3. Train a model:")
    print("   python src/train.py --model resnet50 --epochs 50")
    print()
    print("4. Evaluate the model:")
    print("   python src/evaluate.py \\")
    print("      --model data/models/resnet50/resnet50_best.keras \\")
    print("      --encoder data/models/resnet50/label_encoder.pkl \\")
    print("      --test-data data/processed")
    print()
    print("5. Run Streamlit app:")
    print("   streamlit run app/streamlit_app.py")
    print()
    print("6. Real-time webcam recognition:")
    print("   python app/webcam_recognition.py \\")
    print("      --model data/models/resnet50/resnet50_best.keras \\")
    print("      --encoder data/models/resnet50/label_encoder.pkl")
    print()
    print("📚 For more information, see README.md")
    print()


def main():
    """Main setup routine"""
    print_header("Face Classification Project Setup")
    print("This script will set up your development environment")
    
    # Run setup steps
    steps = [
        ("Check Python version", check_python_version),
        ("Create directories", setup_directories),
        ("Install dependencies", install_requirements),
        ("Test imports", test_imports),
        ("Check GPU support", check_gpu),
        ("Create configuration", create_sample_config),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n⚠️  Warning: {step_name} failed")
            print("You may need to fix this manually before proceeding")
    
    # Optional: Download dataset
    download_test_data()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
