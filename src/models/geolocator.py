"""GeoLocator: hierarchical coordinate prediction from satellite embeddings."""

import torch
import torch.nn as nn
import math


class GeoLocator(nn.Module):
    """
    Hierarchical geolocation model.
    Predicts: continent -> country -> region -> grid cell
    Also supports direct coordinate regression with Haversine loss.
    """

    def __init__(
        self,
        embed_dim=384,
        num_continent_classes=7,
        num_country_classes=195,
        num_region_classes=1000,
        num_grid_classes=64800,
        use_hierarchy=True,
        dropout=0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_hierarchy = use_hierarchy

        # Shared feature extractor
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        if use_hierarchy:
            # Hierarchical classifiers
            self.continent_head = nn.Linear(embed_dim, num_continent_classes)
            self.country_head = nn.Linear(embed_dim + num_continent_classes, num_country_classes)
            self.region_head = nn.Linear(embed_dim + num_continent_classes + num_country_classes, num_region_classes)
            self.grid_head = nn.Linear(embed_dim + num_continent_classes + num_country_classes + num_region_classes, num_grid_classes)
        else:
            # Direct coordinate regression
            self.coord_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 2)  # [lat, lon]
            )

    def forward(self, embedding):
        """
        Args:
            embedding: (B, embed_dim) from vision encoder

        Returns:
            If hierarchical: dict with logits at each level
            If regression: (B, 2) coordinates [lat, lon]
        """
        features = self.feature_proj(embedding)

        if self.use_hierarchy:
            # Continent prediction
            continent_logits = self.continent_head(features)
            continent_prob = torch.softmax(continent_logits, dim=-1)

            # Country prediction (conditioned on continent)
            country_input = torch.cat([features, continent_prob], dim=-1)
            country_logits = self.country_head(country_input)
            country_prob = torch.softmax(country_logits, dim=-1)

            # Region prediction (conditioned on country)
            region_input = torch.cat([features, continent_prob, country_prob], dim=-1)
            region_logits = self.region_head(region_input)
            region_prob = torch.softmax(region_logits, dim=-1)

            # Grid prediction (conditioned on region)
            grid_input = torch.cat([features, continent_prob, country_prob, region_prob], dim=-1)
            grid_logits = self.grid_head(grid_input)

            return {
                'continent': continent_logits,
                'country': country_logits,
                'region': region_logits,
                'grid': grid_logits
            }
        else:
            coords = self.coord_head(features)
            # Constrain to valid ranges: lat [-90, 90], lon [-180, 180]
            coords[:, 0] = torch.tanh(coords[:, 0]) * 90  # latitude
            coords[:, 1] = torch.tanh(coords[:, 1]) * 180  # longitude
            return coords


def haversine_loss(pred_coords, true_coords):
    """
    Haversine distance loss for coordinate regression.

    Args:
        pred_coords: (B, 2) [lat, lon] in degrees
        true_coords: (B, 2) [lat, lon] in degrees

    Returns:
        Mean haversine distance in kilometers
    """
    # Convert to radians
    pred_rad = torch.deg2rad(pred_coords)
    true_rad = torch.deg2rad(true_coords)

    dlat = pred_rad[:, 0] - true_rad[:, 0]
    dlon = pred_rad[:, 1] - true_rad[:, 1]

    a = torch.sin(dlat / 2) ** 2 +         torch.cos(true_rad[:, 0]) * torch.cos(pred_rad[:, 0]) * torch.sin(dlon / 2) ** 2

    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Earth radius in km
    R = 6371.0
    distance = R * c

    return distance.mean()


def grid_to_coordinates(grid_indices, grid_size=1.0):
    """
    Convert grid cell indices to approximate coordinates.

    Grid layout: 360 rows (latitude -90 to 90) x 180 columns (longitude -180 to 180)
    Actually: 180 lat divisions x 360 lon divisions = 64800 cells for 1° x 1°

    Args:
        grid_indices: (B,) cell indices
        grid_size: size of grid cell in degrees (default 1.0)

    Returns:
        (B, 2) [lat, lon] center coordinates of cells
    """
    num_lon_cells = int(360 / grid_size)  # 360
    num_lat_cells = int(180 / grid_size)  # 180

    # Convert index to lat/lon grid position
    lat_idx = grid_indices // num_lon_cells  # 0 to 179 (from -90)
    lon_idx = grid_indices % num_lon_cells   # 0 to 359 (from -180)

    # Convert to coordinates (center of cell)
    lat = -90 + (lat_idx.float() + 0.5) * grid_size
    lon = -180 + (lon_idx.float() + 0.5) * grid_size

    return torch.stack([lat, lon], dim=-1)
