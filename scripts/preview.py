import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so src package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.query import get_file_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-bright',    type=float, default=100.0)
    parser.add_argument('--max-entropy',   type=float, default=4.5)
    parser.add_argument('--min-contrast',  type=float, default=None)
    parser.add_argument('--min-sharpness', type=float, default=None)
    parser.add_argument('--limit',         type=int,   default=9)
    args = parser.parse_args()

    paths = get_file_list(
        min_bright=args.min_bright,
        max_entropy=args.max_entropy,
        min_contrast=args.min_contrast,
        min_sharpness=args.min_sharpness,
        limit=args.limit
    )

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, p in zip(axes.flatten(), paths):
        img = Image.open(p)
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()