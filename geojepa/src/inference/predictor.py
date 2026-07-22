"""Inference pipeline for GeoJEPA."""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image

from ..models.geojepa import GeoJEPA
from ..models.geolocator import GeoLocator, grid_to_coordinates
from ..models.geodescriber import GeoDescriber
from ..config import DEFAULT_CONFIG


class GeoJEPAPredictor:
    """End-to-end predictor for geolocation and description."""

    def __init__(self, checkpoint_path, config=None, device=None):
        self.config = config or DEFAULT_CONFIG
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = GeoJEPA(self.config).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load downstream heads if available
        self.model.add_downstream_heads()
        heads_path = Path(checkpoint_path).parent / "downstream_heads.pt"
        if heads_path.exists():
            heads = torch.load(heads_path, map_location=self.device)
            self.model.geo_locator.load_state_dict(heads["locator"])
            self.model.geo_describer.load_state_dict(heads["describer"])

        self.model.geo_locator.to(self.device).eval()
        self.model.geo_describer.to(self.device).eval()

        # Image preprocessing
        self.image_size = self.config.data.image_size
        self.bands = self.config.data.bands
        self.reflectance_scale = self.config.data.reflectance_scale

    def preprocess_image(self, image_path):
        """Load and preprocess satellite image."""
        # Load image
        if isinstance(image_path, (str, Path)):
            img = Image.open(image_path).convert("RGB")
        else:
            img = image_path

        # Resize
        img = img.resize((self.image_size, self.image_size))

        # To tensor
        image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

        # Add NIR if needed
        if len(self.bands) == 4 and image.shape[0] == 3:
            nir = image[0] * 0.3 + image[1] * 0.3 + image[2] * 0.4
            image = torch.cat([image, nir.unsqueeze(0)], dim=0)

        # Normalize
        image = image / 255.0  # Assuming 8-bit image

        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)

        return image

    def predict(self, image_path, return_description=True):
        """
        Predict location and description from satellite image.

        Args:
            image_path: path to satellite image
            return_description: whether to generate text description

        Returns:
            dict with predictions
        """
        image = self.preprocess_image(image_path)

        with torch.no_grad():
            # Get embedding
            embedding = self.model.get_embedding(image)

            # Geolocation
            loc_outputs = self.model.geo_locator(embedding)

            # Get predicted coordinates from grid
            pred_grid = torch.argmax(loc_outputs["grid"], dim=-1)
            pred_coords = grid_to_coordinates(pred_grid)

            # Get probabilities for each level
            continent_prob = F.softmax(loc_outputs["continent"], dim=-1)
            country_prob = F.softmax(loc_outputs["country"], dim=-1)
            region_prob = F.softmax(loc_outputs["region"], dim=-1)
            grid_prob = F.softmax(loc_outputs["grid"], dim=-1)

            result = {
                "coordinates": {
                    "latitude": pred_coords[0, 0].item(),
                    "longitude": pred_coords[0, 1].item()
                },
                "confidence": {
                    "continent": continent_prob.max().item(),
                    "country": country_prob.max().item(),
                    "region": region_prob.max().item(),
                    "grid": grid_prob.max().item()
                },
                "hierarchy": {
                    "continent_id": continent_prob.argmax().item(),
                    "country_id": country_prob.argmax().item(),
                    "region_id": region_prob.argmax().item(),
                    "grid_id": pred_grid.item()
                }
            }

            # Generate description
            if return_description:
                description = self.model.geo_describer.generate(
                    embedding,
                    pred_coords,
                    max_length=128
                )
                result["description"] = description

        return result

    def predict_batch(self, image_paths):
        """Predict for batch of images."""
        images = torch.cat([self.preprocess_image(p) for p in image_paths])

        with torch.no_grad():
            embeddings = self.model.get_embedding(images)
            loc_outputs = self.model.geo_locator(embeddings)

            pred_grids = torch.argmax(loc_outputs["grid"], dim=-1)
            pred_coords = grid_to_coordinates(pred_grids)

        results = []
        for i in range(len(image_paths)):
            results.append({
                "coordinates": {
                    "latitude": pred_coords[i, 0].item(),
                    "longitude": pred_coords[i, 1].item()
                }
            })

        return results

    def get_embedding(self, image_path):
        """Get raw embedding for an image."""
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            return self.model.get_embedding(image)


# Continent names for human-readable output
CONTINENT_NAMES = {
    0: "Europe/Asia",
    1: "Africa",
    2: "Asia",
    3: "North America",
    4: "South America",
    5: "Australia/Oceania",
    6: "Antarctica"
}


def format_prediction(result):
    """Format prediction result for display."""
    coords = result["coordinates"]
    conf = result["confidence"]
    hier = result["hierarchy"]

    output = []
    output.append("=" * 50)
    output.append("GEOLOCATION RESULT")
    output.append("=" * 50)
    output.append(f"Predicted Coordinates: {coords['latitude']:.4f}°, {coords['longitude']:.4f}°")
    output.append("")
    output.append("Hierarchical Prediction:")
    output.append(f"  Continent: {CONTINENT_NAMES.get(hier['continent_id'], 'Unknown')} (conf: {conf['continent']:.2%})")
    output.append(f"  Country ID: {hier['country_id']} (conf: {conf['country']:.2%})")
    output.append(f"  Region ID: {hier['region_id']} (conf: {conf['region']:.2%})")
    output.append(f"  Grid Cell: {hier['grid_id']} (conf: {conf['grid']:.2%})")
    output.append("")

    if "description" in result:
        output.append("Description:")
        output.append(f"  {result['description']}")
        output.append("")

    output.append("=" * 50)

    return "\n".join(output)
