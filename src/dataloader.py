from torch.utils.data import Dataset
from PIL import Image

class MetadataFilteredDataset(Dataset):
    def __init__(self, data, transform=None):
        # data: list of either file_path strings or (file_path, label) tuples
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item, tuple):
            path, label = item
        else:
            path, label = item, None
        # Open image from file path
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Return (img, label) if label exists, else just img
        return (img, label) if label is not None else img
