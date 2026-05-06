"""
Dataset Download and Preparation Script
Downloads LFW dataset and prepares for training
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_lfw_dataset(data_dir="data/raw"):
    """
    Download Labeled Faces in the Wild dataset
    
    Args:
        data_dir: Directory to store dataset
    """
    print("=" * 70)
    print("Downloading Labeled Faces in the Wild (LFW) Dataset")
    print("=" * 70)
    
    # Create directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # LFW dataset URL
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    tar_path = data_path / "lfw.tgz"
    
    # Download
    if not tar_path.exists():
        print(f"\n📥 Downloading LFW dataset...")
        print(f"   URL: {lfw_url}")
        print(f"   Size: ~173 MB")
        download_url(lfw_url, str(tar_path))
        print("✅ Download complete!")
    else:
        print(f"\n✅ LFW dataset already downloaded: {tar_path}")
    
    # Extract
    extract_path = data_path / "lfw"
    if not extract_path.exists():
        print(f"\n📦 Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=data_path)
        print(f"✅ Extracted to: {extract_path}")
    else:
        print(f"\n✅ Dataset already extracted: {extract_path}")
    
    # Dataset statistics
    print(f"\n📊 Dataset Statistics:")
    person_dirs = [d for d in extract_path.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob('*.jpg'))) for d in person_dirs)
    
    print(f"   Total people: {len(person_dirs)}")
    print(f"   Total images: {total_images}")
    
    # Filter people with multiple images
    min_images = 10
    filtered_people = []
    filtered_images = 0
    
    for person_dir in person_dirs:
        num_images = len(list(person_dir.glob('*.jpg')))
        if num_images >= min_images:
            filtered_people.append(person_dir.name)
            filtered_images += num_images
    
    print(f"\n   People with ≥{min_images} images: {len(filtered_people)}")
    print(f"   Images for these people: {filtered_images}")
    
    print("\n" + "=" * 70)
    print("Dataset Ready!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Run face detection: python src/preprocessing/face_detection.py \\")
    print(f"      --input {extract_path} --output data/processed")
    print(f"2. Train model: python src/train.py --model resnet50")
    
    return str(extract_path)


def prepare_sample_dataset(source_dir="data/raw/lfw", output_dir="data/sample",
                          num_people=20, min_images=15):
    """
    Create a smaller sample dataset for quick testing
    
    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for sample
        num_people: Number of people to include
        min_images: Minimum images per person
    """
    print("\n" + "=" * 70)
    print("Creating Sample Dataset")
    print("=" * 70)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get people with enough images
    person_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    eligible_people = []
    
    for person_dir in person_dirs:
        num_images = len(list(person_dir.glob('*.jpg')))
        if num_images >= min_images:
            eligible_people.append(person_dir)
    
    # Select random sample
    import random
    random.seed(42)
    selected_people = random.sample(eligible_people, min(num_people, len(eligible_people)))
    
    # Copy selected people
    total_images = 0
    print(f"\nCopying {len(selected_people)} people...")
    
    for person_dir in tqdm(selected_people):
        dest_dir = output_path / person_dir.name
        dest_dir.mkdir(exist_ok=True)
        
        # Copy images
        for img_file in person_dir.glob('*.jpg'):
            shutil.copy2(img_file, dest_dir / img_file.name)
            total_images += 1
    
    print(f"\n✅ Sample dataset created!")
    print(f"   Location: {output_path}")
    print(f"   People: {len(selected_people)}")
    print(f"   Images: {total_images}")
    
    return str(output_path)


def download_dataset(dataset_name="lfw", data_dir="data/raw"):
    """
    Download specified dataset
    
    Args:
        dataset_name: Dataset to download (lfw, vggface2)
        data_dir: Directory to store dataset
    """
    if dataset_name.lower() == "lfw":
        return download_lfw_dataset(data_dir)
    elif dataset_name.lower() == "vggface2":
        print("\n⚠️  VGGFace2 dataset requires manual download")
        print("   Visit: https://github.com/ox-vgg/vgg_face2")
        print("   After downloading, extract to:", data_dir)
        return None
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download face dataset')
    parser.add_argument('--dataset', type=str, default='lfw',
                       choices=['lfw', 'vggface2'],
                       help='Dataset to download')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory to store dataset')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create small sample dataset for testing')
    
    args = parser.parse_args()
    
    # Download dataset
    dataset_path = download_dataset(args.dataset, args.data_dir)
    
    # Create sample if requested
    if args.create_sample and dataset_path:
        prepare_sample_dataset(
            source_dir=dataset_path,
            output_dir="data/sample",
            num_people=20,
            min_images=15
        )
