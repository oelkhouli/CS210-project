import os
import sys
import pickle
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# Ensure project root is on sys.path so src package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.query import get_file_list
from src.dataloader import MetadataFilteredDataset

# Helper to load CIFAR-10 labels from pickled batches
def load_labels(root_dir='data/raw'):
    labels = {}
    for fname in os.listdir(root_dir):
        if fname.startswith('data_batch') and fname.endswith(('.pkl', '.pickle')):
            with open(os.path.join(root_dir, fname), 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                files, lbs = batch[b'filenames'], batch[b'labels']
                for fn, lb in zip(files, lbs):
                    base = os.path.basename(fn.decode()) + '.png'
                    labels[os.path.join('data/raw/train', base)] = lb
    return labels

# Prepare train/validation datasets based on metadata filters and optional limit
def prepare_datasets(min_bright, max_entropy, limit, val_split, seed):
    # Fetch paths using metadata filters
    paths = get_file_list(
        min_bright=args.min_bright,
        max_entropy=args.max_entropy,
        min_contrast=args.min_contrast,
        min_sharpness=args.min_sharpness,
        limit=args.limit
    )
    labels_map = load_labels('data/raw')
    data = [(p, labels_map.get(p, 0)) for p in paths]
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_split))
    train_data, val_data = data[:split_idx], data[split_idx:]
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    train_ds = MetadataFilteredDataset(train_data, transform=transform)
    val_ds = MetadataFilteredDataset(val_data, transform=transform)
    return train_ds, val_ds

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 8 * 8, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Main training routine
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on CIFAR-10 subset filtered by metadata")
    parser.add_argument('--min-bright',    type=float, default=100.0, help='Minimum brightness filter')
    parser.add_argument('--max-entropy',   type=float, default=4.5,   help='Maximum entropy filter')
    parser.add_argument('--min-contrast',  type=float, default=None,  help='Minimum contrast filter')
    parser.add_argument('--min-sharpness', type=float, default=None,  help='Minimum sharpness filter')
    parser.add_argument('--limit',         type=int,   default=None,  help='Limit number of images (for debug)')
    parser.add_argument('--val-split',     type=float, default=0.2,   help='Validation split fraction')
    parser.add_argument('--batch-size',    type=int,   default=64,    help='Batch size')
    parser.add_argument('--epochs',        type=int,   default=5,     help='Training epochs')
    parser.add_argument('--lr',            type=float, default=1e-3,  help='Learning rate')
    parser.add_argument('--seed',          type=int,   default=42,    help='Random seed')
    parser.add_argument('--device',        type=str,   default='cpu',  help='Device: cpu or cuda')
    args = parser.parse_args()

    # Select device
    device = torch.device('cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_ds, val_ds = prepare_datasets(
        args.min_bright, args.max_entropy, args.limit, args.val_split, args.seed
    )
    print(f"Loaded datasets: train={len(train_ds)} images, val={len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # Model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training/validation loop
    for epoch in range(1, args.epochs+1):
        # Training phase
        model.train()
        train_loss = 0.0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labs)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labs).sum().item()
                total += labs.size(0)
        val_loss /= len(val_loader)
        acc = 100 * correct / total if total>0 else 0.0

        print(f"Epoch {epoch}/{args.epochs} — "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {acc:.2f}%")
