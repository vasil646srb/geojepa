import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class GeoImageDataset(Dataset):
    def __init__(self, data_dir="data/hls_real", transform=None):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("sample_*.npz"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = torch.from_numpy(data['image']).float()
        image = (image - 0.5) / 0.5
        lat = float(data['lat'])
        lon = float(data['lon'])
        coords = torch.tensor([lat, lon], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, coords
