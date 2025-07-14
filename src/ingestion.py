import os
import argparse
from torchvision.datasets import CIFAR10
from PIL import Image

# Entry point for exporting CIFAR-10 images
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Export CIFAR-10 images as PNGs organized by class."
    )
    parser.add_argument(
        '--root_dir', default='data/raw',
        help='Root directory to store raw images (will create a train/ subfolder).'
    )
    args = parser.parse_args()

    # Ensure we are in project root
    cwd = os.getcwd()
    print(f"[ingestion.py] Running from working directory: {cwd}")
    print(f"[ingestion.py] Target root_dir: {args.root_dir}\n")

    # Prepare output directories
    train_dir = os.path.join(args.root_dir, 'train')
    try:
        os.makedirs(train_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {train_dir}: {e}")
        exit(1)

    # Download CIFAR-10
    print("[ingestion.py] Downloading CIFAR-10 dataset...")
    try:
        ds = CIFAR10(args.root_dir, train=True, download=True)
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")
        exit(1)
    print("[ingestion.py] Download complete. Exporting images...")

    # Export images
    for idx, (img, label) in enumerate(ds):
        class_name = ds.classes[label]
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f'{idx}.png')
        img.save(img_path)
        if (idx + 1) % 1000 == 0:
            print(f"[ingestion.py] Saved {idx + 1} images so far...")

    print(f"[ingestion.py] Export complete. All images saved under: {train_dir}")
