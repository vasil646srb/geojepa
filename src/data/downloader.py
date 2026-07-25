"""Automatic satellite imagery downloader for GeoJEPA training."""

import os
import time
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    from sentinelhub import (
        SentinelHubRequest, DataCollection, MimeType, CRS, BBox,
        SHConfig, SentinelHubDownloadClient, bbox_to_dimensions
    )
    SENTINELHUB_AVAILABLE = True
except ImportError:
    SENTINELHUB_AVAILABLE = False
    print("Warning: sentinelhub not installed. Use: pip install sentinelhub")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class DownloadConfig:
    """Configuration for data download."""
    output_dir: str = "data/satellite"
    image_size: int = 256  # pixels
    resolution: int = 10   # meters per pixel (Sentinel-2 native: 10m)
    max_cloud_coverage: float = 20.0  # percent
    bands: List[str] = None
    time_range: Tuple[str, str] = ("2023-01-01", "2024-01-01")
    max_samples: int = 100000

    # Sentinel Hub credentials (get from https://dataspace.copernicus.eu/)
    sh_client_id: str = None
    sh_client_secret: str = None

    def __post_init__(self):
        if self.bands is None:
            self.bands = ["B04", "B03", "B02", "B08"]  # RGB + NIR
        os.makedirs(self.output_dir, exist_ok=True)


class SatelliteDataDownloader:
    """Downloads Sentinel-2 imagery from Copernicus Data Space Ecosystem."""

    # Predefined regions for balanced global coverage
    REGIONS = [
        # Europe
        {"name": "europe_west", "bbox": (-10, 35, 15, 55), "weight": 1.0},
        {"name": "europe_east", "bbox": (15, 45, 40, 60), "weight": 1.0},
        # North America
        {"name": "na_east", "bbox": (-100, 25, -60, 50), "weight": 1.0},
        {"name": "na_west", "bbox": (-130, 30, -100, 50), "weight": 1.0},
        # South America
        {"name": "sa_north", "bbox": (-80, -10, -35, 10), "weight": 1.0},
        {"name": "sa_south", "bbox": (-75, -55, -35, -20), "weight": 0.8},
        # Africa
        {"name": "africa_north", "bbox": (-20, 0, 40, 25), "weight": 1.0},
        {"name": "africa_central", "bbox": (10, -15, 40, 10), "weight": 1.0},
        {"name": "africa_south", "bbox": (15, -35, 35, -15), "weight": 0.8},
        # Asia
        {"name": "asia_east", "bbox": (100, 20, 145, 50), "weight": 1.0},
        {"name": "asia_central", "bbox": (50, 25, 100, 50), "weight": 1.0},
        {"name": "asia_south", "bbox": (65, 5, 95, 35), "weight": 1.0},
        # Australia
        {"name": "australia", "bbox": (110, -40, 155, -10), "weight": 0.8},
        # Special terrains
        {"name": "sahara", "bbox": (-15, 15, 35, 30), "weight": 0.5},
        {"name": "amazon", "bbox": (-75, -15, -45, 5), "weight": 0.5},
        {"name": "siberia", "bbox": (60, 50, 140, 70), "weight": 0.5},
        {"name": "himalaya", "bbox": (70, 25, 100, 40), "weight": 0.5},
    ]

    def __init__(self, config: DownloadConfig = None):
        self.config = config or DownloadConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata_file = self.output_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # Initialize Sentinel Hub
        if SENTINELHUB_AVAILABLE and self.config.sh_client_id:
            self.config_sh = SHConfig()
            self.config_sh.sh_client_id = self.config.sh_client_id
            self.config_sh.sh_client_secret = self.config.sh_client_secret
            self.config_sh.sh_base_url = "https://sh.dataspace.copernicus.eu"
            self.config_sh.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        else:
            self.config_sh = None

    def _load_metadata(self):
        """Load existing metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"samples": [], "total_downloaded": 0}

    def _save_metadata(self):
        """Save metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _random_bbox(self, region_bbox, size_km=25.6):
        """Generate random bbox within region."""
        min_lon, min_lat, max_lon, max_lat = region_bbox

        # Convert degrees to approx km (rough estimate)
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        # Size in degrees (256px * 10m = 2.56km)
        size_deg_lat = size_km / 111.0  # 1 deg lat ~ 111 km
        size_deg_lon = size_km / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))

        # Random position
        lat = random.uniform(min_lat, max_lat - size_deg_lat)
        lon = random.uniform(min_lon, max_lon - size_deg_lon)

        return (lon, lat, lon + size_deg_lon, lat + size_deg_lat)

    def _get_evalscript(self):
        """Generate evalscript for requested bands."""
        bands = self.config.bands
        band_list = ", ".join([f'"{b}"' for b in bands])

        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: [{{
                    bands: [{band_list}],
                    units: "DN"
                }}],
                output: {{
                    bands: {len(bands)},
                    sampleType: "UINT16"
                }}
            }};
        }}

        function evaluatePixel(sample) {{
            return [{', '.join([f'sample.{b}' for b in bands])}];
        }}
        """

    def download_sample_sentinelhub(self, bbox, filename):
        """Download single sample via Sentinel Hub Process API."""
        if not SENTINELHUB_AVAILABLE or not self.config_sh:
            return False

        try:
            # Calculate dimensions for 10m resolution
            # 256 pixels * 10m = 2560m = 2.56km
            size = bbox_to_dimensions(BBox(bbox, CRS.WGS84), resolution=self.config.resolution)

            # Ensure size is reasonable
            size = (min(size[0], 512), min(size[1], 512))

            request = SentinelHubRequest(
                evalscript=self._get_evalscript(),
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            name="s2l2a",
                            service_url="https://sh.dataspace.copernicus.eu"
                        ),
                        time_interval=self.config.time_range,
                        other_args={{
                            "dataFilter": {{
                                "mosaickingOrder": "leastCC",
                                "maxCloudCoverage": self.config.max_cloud_coverage
                            }}
                        }}
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=BBox(bbox, CRS.WGS84),
                size=size,
                config=self.config_sh
            )

            data = request.get_data()[0]

            # Save as numpy array
            np.save(filename, data)

            return True

        except Exception as e:
            print(f"Download failed for {bbox}: {e}")
            return False

    def download_sample_openeo(self, bbox, filename):
        """Alternative: download via OpenEO (simpler auth)."""
        try:
            import openeo

            # Connect to CDSE
            connection = openeo.connect("openeo.dataspace.copernicus.eu")
            connection.authenticate_oidc()

            # Load collection
            s2 = connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent={
                    "west": bbox[0], "south": bbox[1],
                    "east": bbox[2], "north": bbox[3]
                },
                temporal_extent=[self.config.time_range[0], self.config.time_range[1]],
                bands=self.config.bands,
                max_cloud_cover=self.config.max_cloud_coverage
            )

            # Download
            job = s2.create_job()
            result = job.start_and_wait()
            result.download(filename)

            return True

        except Exception as e:
            print(f"OpenEO download failed: {e}")
            return False

    def download_sample_fallback(self, bbox, filename):
        """Fallback: use Copernicus OData API directly."""
        # This requires manual token management
        # For now, return False to use synthetic data
        return False

    def download_batch(self, num_samples: int = None, use_synthetic_fallback: bool = True):
        """
        Download batch of satellite images.

        Args:
            num_samples: number of samples to download (default: config.max_samples)
            use_synthetic_fallback: if True, generates synthetic data when API fails

        Returns:
            number of successfully downloaded samples
        """
        num_samples = num_samples or self.config.max_samples

        print(f"Starting download of {num_samples} samples...")
        print(f"Output directory: {self.output_dir}")

        downloaded = 0
        failed = 0

        # Weighted region selection
        weights = [r["weight"] for r in self.REGIONS]

        for i in range(num_samples):
            # Select region
            region = random.choices(self.REGIONS, weights=weights)[0]

            # Generate random bbox
            bbox = self._random_bbox(region["bbox"])
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2

            filename = self.output_dir / f"sample_{i:06d}.npy"

            # Try Sentinel Hub
            success = False
            if self.config_sh:
                success = self.download_sample_sentinelhub(bbox, filename)

            # Try OpenEO
            if not success:
                success = self.download_sample_openeo(bbox, filename)

            # Fallback to synthetic
            if not success and use_synthetic_fallback:
                success = self._generate_synthetic_sample(bbox, filename)

            if success:
                # Save metadata
                self.metadata["samples"].append({
                    "image": str(filename),
                    "lat": center_lat,
                    "lon": center_lon,
                    "region": region["name"],
                    "bbox": bbox,
                    "source": "sentinelhub" if self.config_sh else "synthetic"
                })
                downloaded += 1

                if downloaded % 100 == 0:
                    self._save_metadata()
                    print(f"Downloaded: {downloaded}/{num_samples} (failed: {failed})")
            else:
                failed += 1

            # Rate limiting
            if i % 10 == 0:
                time.sleep(0.5)

        self._save_metadata()
        print(f"\nComplete! Downloaded: {downloaded}, Failed: {failed}")
        print(f"Metadata saved to: {self.metadata_file}")

        return downloaded

    def _generate_synthetic_sample(self, bbox, filename):
        """Generate synthetic sample when API is unavailable."""
        from ..data.dataset import SyntheticSatelliteDataset

        # Create temporary dataset for one sample
        dataset = SyntheticSatelliteDataset(num_samples=1, image_size=self.config.image_size)
        sample = dataset[0]

        # Override coordinates with actual bbox center
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2

        # Save
        np.save(filename, sample["image"].numpy())

        # Update metadata
        self.metadata["samples"].append({
            "image": str(filename),
            "lat": center_lat,
            "lon": center_lon,
            "region": "synthetic",
            "bbox": bbox,
            "source": "synthetic"
        })

        return True

    def verify_dataset(self):
        """Verify downloaded dataset integrity."""
        print("Verifying dataset...")

        valid = 0
        invalid = 0

        for sample in self.metadata["samples"]:
            path = Path(sample["image"])
            if not path.exists():
                invalid += 1
                continue

            try:
                data = np.load(path)
                if data.shape[0] != len(self.config.bands):
                    invalid += 1
                else:
                    valid += 1
            except Exception:
                invalid += 1

        print(f"Valid samples: {valid}, Invalid: {invalid}")
        return valid, invalid


def download_satellite_data(
    output_dir="data/satellite",
    num_samples=10000,
    sh_client_id=None,
    sh_client_secret=None,
    use_synthetic=True
):
    """Convenience function to download satellite data."""
    config = DownloadConfig(
        output_dir=output_dir,
        max_samples=num_samples,
        sh_client_id=sh_client_id,
        sh_client_secret=sh_client_secret
    )

    downloader = SatelliteDataDownloader(config)

    # Check if we already have enough data
    existing = len(downloader.metadata["samples"])
    if existing >= num_samples:
        print(f"Already have {existing} samples. Skipping download.")
        return existing

    # Download remaining
    to_download = num_samples - existing
    return downloader.download_batch(to_download, use_synthetic_fallback=use_synthetic)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/satellite")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--sh-client-id", default=None)
    parser.add_argument("--sh-client-secret", default=None)
    parser.add_argument("--no-synthetic", action="store_true")

    args = parser.parse_args()

    download_satellite_data(
        output_dir=args.output,
        num_samples=args.num_samples,
        sh_client_id=args.sh_client_id,
        sh_client_secret=args.sh_client_secret,
        use_synthetic=not args.no_synthetic
    )
