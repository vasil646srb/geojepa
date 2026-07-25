"""Data loading with automatic download capability."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json

from .downloader import download_satellite_data


class SatelliteDataset(Dataset):
    """
    Dataset for satellite imagery with coordinates.
    Auto-downloads data if not present.
    """

    def __init__(
        self,
        data_dir="data/satellite",
        metadata_file="metadata.json",
        image_size=256,
        bands=["B04", "B03", "B02", "B08"],
        normalize=True,
        augment=True,
        auto_download=True,
        num_samples=10000,
        sh_client_id=None,
        sh_client_secret=None
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.bands = bands
        self.normalize = normalize
        self.augment = augment

        # Auto-download if needed
        if auto_download:
            download_satellite_data(
                output_dir=data_dir,
                num_samples=num_samples,
                sh_client_id=sh_client_id,
                sh_client_secret=sh_client_secret,
                use_synthetic=True  # Fallback to synthetic if API unavailable
            )

        # Load metadata
        metadata_path = self.data_dir / metadata_file
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"No metadata found at {metadata_path}. Run with auto_download=True.")

        self.samples = self.metadata.get("samples", [])

        # Normalization
        self.reflectance_scale = 10000.0
        self.band_means = {
            "B02": 1094.0, "B03": 1110.0, "B04": 1250.0, "B08": 2310.0
        }
        self.band_stds = {
            "B02": 760.0, "B03": 780.0, "B04": 960.0, "B08": 1110.0
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img_path = Path(sample["image"])

        if img_path.suffix == ".npy":
            image = torch.from_numpy(np.load(img_path)).float()
            # Ensure correct shape (C, H, W)
            if image.dim() == 3 and image.shape[0] not in [3, 4]:
                image = image.permute(2, 0, 1)
        else:
            # Fallback: load as image
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

            if len(self.bands) == 4 and image.shape[0] == 3:
                nir = image[0] * 0.3 + image[1] * 0.3 + image[2] * 0.4
                image = torch.cat([image, nir.unsqueeze(0)], dim=0)

        # Resize if needed
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear"
            ).squeeze(0)

        # Normalize
        if self.normalize:
            image = image / self.reflectance_scale
            for i, band in enumerate(self.bands):
                if band in self.band_means:
                    image[i] = (image[i] - self.band_means[band] / self.reflectance_scale) /                                (self.band_stds[band] / self.reflectance_scale)

        # Augment
        if self.augment:
            image = self._augment(image)

        return {
            "image": image,
            "coordinates": torch.tensor([sample["lat"], sample["lon"]], dtype=torch.float32),
            "elevation": torch.tensor(sample.get("elevation", 0.0), dtype=torch.float32)
        }

    def _augment(self, image):
        """Apply random augmentations."""
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        image = torch.rot90(image, k, dims=[1, 2])
        if torch.rand(1) > 0.5:
            factor = torch.rand(1) * 0.4 + 0.8
            image = image * factor
        if torch.rand(1) > 0.7:
            noise = torch.randn_like(image) * 0.01
            image = image + noise
        if torch.rand(1) > 0.8:
            cloud_mask = torch.rand_like(image[0]) > 0.3
            image = image * cloud_mask.unsqueeze(0).float()
        return image


class SyntheticSatelliteDataset(Dataset):
    """Synthetic dataset for testing without real data."""

    BIOME_PATTERNS = {
        "forest": {"color": [0.1, 0.4, 0.1], "texture": "noise"},
        "desert": {"color": [0.8, 0.7, 0.5], "texture": "smooth"},
        "water": {"color": [0.1, 0.2, 0.5], "texture": "smooth"},
        "urban": {"color": [0.5, 0.5, 0.5], "texture": "grid"},
        "snow": {"color": [0.9, 0.9, 0.95], "texture": "noise"},
        "agriculture": {"color": [0.6, 0.7, 0.2], "texture": "striped"},
    }

    def __init__(self, num_samples=10000, image_size=256, num_bands=4):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_bands = num_bands

        np.random.seed(42)
        self.coordinates = []
        for _ in range(num_samples):
            lat = np.random.uniform(-60, 70)
            lon = np.random.uniform(-180, 180)
            self.coordinates.append((lat, lon))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        lat, lon = self.coordinates[idx]

        if lat > 60:
            biome = "snow"
        elif lat > 35:
            biome = "forest" if np.random.rand() > 0.3 else "agriculture"
        elif lat > -10:
            biome = "forest" if np.random.rand() > 0.5 else "urban" if np.random.rand() > 0.7 else "agriculture"
        elif lat > -35:
            biome = "desert" if np.random.rand() > 0.5 else "forest"
        else:
            biome = "snow"

        image = self._generate_biome_image(biome)

        return {
            "image": image,
            "coordinates": torch.tensor([lat, lon], dtype=torch.float32),
            "elevation": torch.tensor(np.random.uniform(0, 3000), dtype=torch.float32)
        }

    def _generate_biome_image(self, biome):
        pattern = self.BIOME_PATTERNS[biome]
        h, w = self.image_size, self.image_size
        base = torch.tensor(pattern["color"]).view(3, 1, 1).expand(3, h, w)

        if pattern["texture"] == "noise":
            texture = torch.randn(3, h, w) * 0.1
        elif pattern["texture"] == "grid":
            grid = torch.zeros(3, h, w)
            grid[:, ::8, :] = 0.2
            grid[:, :, ::8] = 0.2
            texture = grid
        elif pattern["texture"] == "striped":
            stripes = torch.zeros(3, h, w)
            stripes[:, ::16, :] = 0.15
            texture = stripes
        else:
            texture = torch.zeros(3, h, w)

        image = torch.clamp(base + texture, 0, 1)

        if self.num_bands == 4:
            nir = image[0] * 0.3 + image[1] * 0.3 + image[2] * 0.4 + torch.randn(h, w) * 0.05
            image = torch.cat([image, nir.unsqueeze(0)], dim=0)

        return image


def create_dataloader(
    data_dir="data/satellite",
    batch_size=64,
    num_workers=4,
    use_synthetic=False,
    num_synthetic_samples=10000,
    auto_download=True,
    sh_client_id=None,
    sh_client_secret=None,
    **dataset_kwargs
):
    """Create dataloader with auto-download."""
    if use_synthetic:
        dataset = SyntheticSatelliteDataset(num_samples=num_synthetic_samples)
    else:
        dataset = SatelliteDataset(
            data_dir=data_dir,
            auto_download=auto_download,
            num_samples=num_synthetic_samples,
            sh_client_id=sh_client_id,
            sh_client_secret=sh_client_secret,
            **dataset_kwargs
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
