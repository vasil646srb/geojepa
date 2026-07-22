"""Configuration for GeoJEPA project."""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class DataConfig:
    """Data configuration."""
    # Sentinel-2 bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
    bands: List[str] = None
    image_size: int = 256
    patch_size: int = 16
    num_patches: int = 256  # (256/16)^2

    # Normalization (Sentinel-2 reflectance 0-10000)
    reflectance_scale: float = 10000.0

    # Augmentations
    use_augmentation: bool = True
    max_rotation: float = 15.0
    max_cloud_coverage: float = 0.3

    def __post_init__(self):
        if self.bands is None:
            self.bands = ["B04", "B03", "B02", "B08"]


@dataclass  
class ModelConfig:
    """Model architecture configuration."""
    # ViT Encoder
    embed_dim: int = 384
    num_layers: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # JEPA Predictor
    predictor_embed_dim: int = 384
    predictor_num_layers: int = 12
    predictor_num_heads: int = 6

    # Masking
    mask_ratio: float = 0.75  # 75% patches masked

    # GeoLocator
    num_continent_classes: int = 7
    num_country_classes: int = 195
    num_region_classes: int = 1000
    num_grid_classes: int = 64800  # 1° x 1° grid

    # GeoDescriber
    llm_model_name: str = "gpt2"
    max_description_length: int = 256

    # Training
    ema_decay: float = 0.996


@dataclass
class TrainingConfig:
    """Training configuration."""
    # General
    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 100
    warmup_epochs: int = 10

    # Optimizer
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)

    # Scheduler
    min_lr: float = 1e-6

    # Loss weights
    jepa_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.1

    # Checkpointing
    save_every: int = 5
    eval_every: int = 1

    # Device
    device: str = "cuda"
    mixed_precision: bool = True


@dataclass
class GeoJEPAConfig:
    """Full project configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()


# Default config
DEFAULT_CONFIG = GeoJEPAConfig()
